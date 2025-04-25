# ros path setup
import sys
import os
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import random

# Add ROS2 paths to Python path
ros_paths = [
	"/opt/ros/jazzy/lib/python3.12/site-packages",
	# Add any other ROS paths you need
]

for path in ros_paths:
	if path not in sys.path:
		sys.path.append(path)

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan, Image, CameraInfo
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from rosgraph_msgs.msg import Clock
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Time
from std_msgs.msg import Header
from std_msgs.msg import ColorRGBA
import rclpy
from rclpy.node import Node
import pybullet as p
import pybullet_data
import numpy as np
import time
import math
import cv2
import os
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped, Twist
from LidarScan import LidarScan, Range
from make_room import create_room
from nav2_msgs.srv import ClearEntireCostmap
from rclpy.executors import MultiThreadedExecutor
from threading import Lock

def add_cube(p, position, size=0.2, color=[1, 0, 0, 1], mass=0):
	"""
	Add a visible cube to the simulation for debugging purposes.
	
	Args:
		p: PyBullet physics client
		position: [x, y, z] position for the cube
		size: Size of the cube (default 0.2m)
		color: RGBA color (default red)
		mass: Mass of the cube (0 = static, >0 = dynamic)
	
	Returns:
		cube_id: ID of the created cube
	"""
	# Create collision shape
	collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size/2, size/2, size/2])
	
	# Create visual shape with color
	visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[size/2, size/2, size/2], rgbaColor=color)
	
	# Create the body
	cube_id = p.createMultiBody(
		baseMass=mass,
		baseCollisionShapeIndex=collision_shape_id,
		baseVisualShapeIndex=visual_shape_id,
		basePosition=position
	)
	
	print(f"Created debug cube at position {position}, ID: {cube_id}")
	return cube_id

def get_sim():
	physicsClient = p.connect(p.DIRECT)  # Connect to the PyBullet physics server
	p.setAdditionalSearchPath(
		pybullet_data.getDataPath()
	)
	p.setGravity(0, 0, -9.81)  # Set gravity (9.81 m/s²)
	p.setPhysicsEngineParameter(enableFileCaching=0)
	p.setRealTimeSimulation(
		0
	)
	p.setTimeStep(1/60.)

	# Load the ground plane
	planeId = p.loadURDF("plane.urdf")
	print(f"Ground plane loaded: {'Success' if planeId >= 0 else 'Failed'}")

	# Create a directional light
	p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
	lightDirection = [0.52, 0.8, 0.7]
	p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 1)
	p.configureDebugVisualizer(p.COV_ENABLE_PLANAR_REFLECTION, 0)

	return p

def get_lidar_data(p, position, orientation, num_rays=600, max_distance=12.0):
	ranges = []

	# Create rotation matrix from quaternion
	# This will transform ray directions from lidar frame to world frame
	rot_matrix = p.getMatrixFromQuaternion(orientation)
	rot_matrix = np.array(rot_matrix).reshape(3, 3)

	for i in range(num_rays):
		# Calculate direction of ray in the lidar's frame (relative to its forward direction)
		angle = 2 * math.pi * i / num_rays
		
		# Direction in lidar frame (forward is x-axis)
		lidar_dir = [math.cos(angle), math.sin(angle), 0]
		
		# Transform direction to world frame using rotation matrix
		world_dir = np.dot(rot_matrix, lidar_dir)
		
		# Cast ray and get hit position
		to_position = [
			position[0] + world_dir[0] * max_distance,
			position[1] + world_dir[1] * max_distance,
			position[2] + world_dir[2] * max_distance,
		]

		ray_result = p.rayTest(position, to_position)[0]
		hit_position = ray_result[3]
		hit_object_uid = ray_result[0]

		# Calculate distance
		if hit_object_uid >= 0:  # If ray hit something
			distance = math.sqrt(
				(hit_position[0] - position[0]) ** 2
				+ (hit_position[1] - position[1]) ** 2
				+ (hit_position[2] - position[2]) ** 2
			)
		else:
			distance = max_distance  # No hit, return max distance

		ranges.append(Range(angle=angle, distance=distance))

	return ranges

def get_camera_data(p, position, orientation):
	width = 320
	height = 240
	fov = 60
	aspect = width / height
	near = 0.02
	far = 50
	# Calculate camera viewpoint
	camera_height = 0.5  # Height above the robot base
	camera_position = [
		position[0],
		position[1],
		position[2] + camera_height,
	]

	# Convert quaternion to Euler angles
	euler = p.getEulerFromQuaternion(orientation)

	# Camera looks in the direction the robot is facing
	yaw = euler[2]
	target_position = [
		camera_position[0] + math.cos(yaw),
		camera_position[1] + math.sin(yaw),
		camera_position[2],
	]

	# Compute view and projection matrices
	view_matrix = p.computeViewMatrix(
		camera_position, target_position, [0, 0, 1]
	)
	projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

	# Get camera image
	images = p.getCameraImage(
		width,
		height,
		view_matrix,
		projection_matrix,
		renderer=p.ER_BULLET_HARDWARE_OPENGL,
	)

	# Extract RGB image
	rgb_array = np.array(images[2], dtype=np.uint8)
	rgb_array = np.reshape(rgb_array, (height, width, 4))
	rgb_array = rgb_array[:, :, :3]  # Remove alpha channel

	return rgb_array

class SimNode(Node):
	def __init__(self):
		super().__init__('sim')
		# Use a callback group to allow concurrent callbacks
		self.callback_group = ReentrantCallbackGroup()

		# Create action client for navigation
		self.nav_client = ActionClient(
			self,
			NavigateToPose,
			'navigate_to_pose',
			callback_group=self.callback_group
		)
		self._sim = get_sim()
		self.timestep = 1/60.
		self.sim_time = 0.

		self.base_frame='base_link'
		self.lidar_frame='lidar_link'
		self.camera_frame='camera_link'

		self.linear_vel = 0.
		self.angular_vel = 1.

		self.tf_broadcaster = TransformBroadcaster(self)
		self.static_tf_broadcaster = StaticTransformBroadcaster(self)
		self._publish_static_transforms()

		self.reset_lock = Lock()

		self._odo_publisher = self.create_publisher(
			Odometry,
			'odom',
			10
		)

		self._odo_publisher = self.create_publisher(
			Odometry,
			'odom',
			10
		)

		self._lidar_publisher = self.create_publisher(
			LaserScan,
			'scan',
			10
		)

		self._camera_publisher = self.create_publisher(
			Image,
			'camera',
			10
		)

		self._camera_info_publisher = self.create_publisher(
			CameraInfo,
			'camera/camera_info',
			10
		)

		self._cmd_vel_subscriber = self.create_subscription(
			Twist,
			'/cmd_vel',
			lambda msg: self._cmd_vel_cb(msg),
			10
		)

		self.marker_publisher = self.create_publisher(
			MarkerArray,
			'goal_markers',
			10
		)

		## Stuff for managing the sim
		self._robot_id = None
		self.obstacle_ids = []
		self.x_bounds = (-7.0, 7.0)
		self.y_bounds = (-7.0, 7.0)

		# Create action client for navigation
		self.nav_client = ActionClient(
			self,
			NavigateToPose,
			'navigate_to_pose',
			callback_group=self.callback_group
		)
		self.clear_global_costmap = self.create_client(
			ClearEntireCostmap,
			'global_costmap/clear_entirely_global_costmap',
			callback_group=self.callback_group
		)
		self.clear_local_costmap = self.create_client(
			ClearEntireCostmap,
			'local_costmap/clear_entirely_local_costmap',
			callback_group=self.callback_group
		)


		# Track current navigation goal
		self.current_goal = None
		self.goal_handle = None
		self.nav_ready = False

		self._timer = self.create_timer(0.01, self.sim_cb)
		self._clock_publisher = self.create_publisher(Clock, '/clock', 10)
		# Create timer for periodic status checking
		#self.timer = self.create_timer(1.0, self.check_nav2_status, callback_group=self.callback_group)

		self._start_task_timer = None
		self.task_running = False
		self.is_resetting = False

		self.reset_simulation_and_nav()

	def publish_goal_marker(self):
		if self.current_goal is None:
			return

		# Create marker for goal position
		marker_array = MarkerArray()
		goal_pose = self.current_goal.pose
		# Goal position marker (red sphere)
		goal_marker = Marker()
		goal_marker.header.frame_id = 'map'
		goal_marker.header.stamp = self.get_time_msg()
		goal_marker.ns = "goal_position"
		goal_marker.pose = goal_pose

		goal_marker.id = 0  # Always use id 0 for the goal
		goal_marker.type = Marker.SPHERE
		goal_marker.action = Marker.ADD
		goal_marker.scale.x = 0.5
		goal_marker.scale.y = 0.5
		goal_marker.scale.z = 0.5
		goal_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
		goal_marker.lifetime.sec = 0  # persists until replaced

		# Goal arrow (green arrow pointing upward)
		arrow_marker = Marker()
		arrow_marker.header.frame_id = 'map'
		arrow_marker.header.stamp = self.get_time_msg()
		arrow_marker.ns = "goal_arrow"
		arrow_marker.id = 0
		arrow_marker.type = Marker.ARROW
		arrow_marker.action = Marker.ADD
		arrow_marker.pose = goal_pose
		arrow_marker.scale.x = 0.7  # arrow length
		arrow_marker.scale.y = 0.1  # arrow width
		arrow_marker.scale.z = 0.1  # arrow height
		arrow_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
		arrow_marker.lifetime.sec = 0  # persists until replaced

		marker_array.markers.append(goal_marker)
		marker_array.markers.append(arrow_marker)

		# Publish the markers
		self.marker_publisher.publish(marker_array)

	def goal_response_callback(self, future):
		"""Callback when goal is accepted or rejected."""
		goal_handle = future.result()

		if not goal_handle.accepted:
			self.get_logger().error('Goal was rejected')
			self.current_goal = None
			self.task_running = False
			# Try again after a short delay
			self.create_timer(5.0, self.start_new_task_once, callback_group=self.callback_group)
			return

		self.get_logger().info('Goal accepted')
		self.goal_handle = goal_handle

		# Get result async
		#result_future = goal_handle.get_result_async()
		#result_future.add_done_callback(self.goal_result_callback)

	def check_nav2_status(self):
		"""Check if Nav2 action server is available."""
		if self.current_goal is not None or self.task_running or self.is_resetting:
			return

		# Check if Nav2 action server is available
		if self.nav_client.wait_for_server(timeout_sec=0.1):
			if not self.nav_ready:
				self.get_logger().info('Nav2 navigation action server is ready')
				self.nav_ready = True

			# If we have services for costmap clearing, we're really ready
			if self.clear_global_costmap.wait_for_service(timeout_sec=0.1):
				self.start_new_task()
			else:
				self.get_logger().info('Waiting for costmap services...')
		else:
			self.nav_ready = False
			self.get_logger().info('Waiting for Nav2 action server...')

	def random_position(self):
		new_x = random.uniform(*self.x_bounds)
		new_y = random.uniform(*self.y_bounds)
		new_theta = random.uniform(-math.pi, math.pi)

		return new_x, new_y, new_theta

	def generate_random_goal(self):
		"""Generate a random navigation goal within the defined bounds."""
		pos, _ = self._robot_pos
		curx = pos[0]
		cury = pos[1]

		new_x, new_y, new_theta = self.random_position()

		goal = PoseStamped()
		goal.header.frame_id = 'map'
		goal.header.stamp = self.get_time_msg()
		goal.pose.position.x = new_x
		goal.pose.position.y = new_y
		goal.pose.position.z = 0.0

		# Convert new_theta to quaternion (yaw only)
		goal.pose.orientation.x = 0.0
		goal.pose.orientation.y = 0.0
		goal.pose.orientation.z = math.sin(new_theta / 2.0)
		goal.pose.orientation.w = math.cos(new_theta / 2.0)

		self.get_logger().info(f'Generated random goal: x={new_x:.2f}, y={new_y:.2f}, theta={new_theta:.2f}')
		return goal

	def send_goal(self, goal_pose):
		"""Send a navigation goal to Nav2."""
		self.get_logger().info('Sending navigation goal...')

		# Wait for action server
		if not self.nav_client.wait_for_server(timeout_sec=2.0):
			self.get_logger().error('Navigation action server not available')
			return False

		# Create goal
		goal_msg = NavigateToPose.Goal()
		goal_msg.pose = goal_pose

		# Send goal
		self.current_goal = goal_pose
		send_goal_future = self.nav_client.send_goal_async(goal_msg)

		# Setup callback for when goal is accepted
		send_goal_future.add_done_callback(self.goal_response_callback)
		return True

	def check_goal_status_and_maybe_update(self):
		with self.reset_lock:
			if self.current_goal is None or not self.task_running:
				return

			goal_x = self.current_goal.pose.position.x
			goal_y = self.current_goal.pose.position.y

			curx, cury = self.robot_xy

			goal = np.array([goal_x, goal_y])
			cur = np.array([curx, cury])

			dist = np.linalg.norm(cur - goal)

			if dist < 1.0:
				self.get_logger().info(f'Goal reached w/in distance {dist}, resetting')
				self.reset_simulation_and_nav()

	def goal_result_callback(self, future):
		"""Callback when goal navigation completes."""
		try:
			result = future.result().result
			status = future.result().status

			if status == 4:  # SUCCEEDED
				self.get_logger().info('Goal reached successfully!')
			else:
				status_map = {
					1: "ABORTED",
					2: "CANCELED",
					3: "UNKNOWN",
					4: "SUCCEEDED"
				}
				status_name = status_map.get(status, f"UNKNOWN ({status})")
				self.get_logger().warning(f'Goal completed with status: {status_name}')

			# Reset and start a new task regardless of outcome
			#self.reset_simulation_and_nav()

		except Exception as e:
			self.get_logger().error(f'Error in goal result callback: {e}')
			self.current_goal = None
			self.goal_handle = None
			# Try again after a short delay
			self.create_timer(5.0, self.start_new_task, callback_group=self.callback_group)

	def reset_simulation_and_nav(self):
		"""Reset both the simulation and Nav2."""
		if self.is_resetting:
			self.get_logger().info('Cowardly ignoring request to reset during reset')

		self.get_logger().info('Resetting simulation and navigation...')
		self.is_resetting = True

		self.task_running = False

		if self.goal_handle is not None:
			try:
				self.get_logger().info('Canceling current goal')
				cancel_future = self.goal_handle.cancel_goal_async()
				rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=1.0)

			except Exception as e:
				self.get_logger().warning('Error canceling goal')

		# Clear flags
		self.current_goal = None
		self.goal_handle = None

		self.reset_pybullet_simulation()

		# Use non-blocking service calls for Nav2 reset
		self.reset_nav2_non_blocking()
		
		# Schedule start_new_task in the next cycle to ensure the reset completes
		self.get_logger().info('Will attempt to start new task in next cycle')

		if hasattr(self, '_start_task_timer') and self._start_task_timer is not None:
			self._start_task_timer.cancel()

		self._start_task_timer = self.create_timer(3., self._reset_complete_callback, callback_group=self.callback_group)

	def reset_pybullet_simulation(self):
		# Clear the simulation
		if self._robot_id is not None:
			self._sim.removeBody(self._robot_id)

		for obs_id in self.obstacle_ids:
			self._sim.removeBody(obs_id)

		self.obstacle_ids = []

		# Load the robot
		#start_x, start_y, start_theta = self.random_position()
		start_x, start_y, start_theta = (0, 0, 0)

		start_pos = [start_x, start_y, 0.1]
		#start_pos = [0, 0, 0.1]
		start_theta = 0.
		start_orientation = p.getQuaternionFromEuler([0, 0, start_theta])
		robot_id = p.loadURDF("husky/husky.urdf", start_pos, start_orientation)

		self._robot_id = robot_id

		# Load obstacles
		wall_ids = create_room(p, size=20)
		n_obstacles = np.random.randint(2, 5)

		for i in range(n_obstacles):
			size = np.random.uniform(0.1, 1.)
			obs_x, obs_y, obs_theta = self.random_position()

			# Give the robot a 1m circle of safety
			obs_dist = np.linalg.norm(
				np.array([obs_x, obs_y]) - np.array([start_x, start_y])
			)
			while obs_dist < 1.:
				obs_x, obs_y, obs_theta = self.random_position()
				obs_dist = np.linalg.norm(
					np.array([obs_x, obs_y]) - np.array([start_x, start_y])
				)

			obs_id = add_cube(
				self._sim,
				position=[obs_x, obs_y],
				size=size,
				mass=100
			)

			self.obstacle_ids.append(obs_id)

		self.obstacle_ids += wall_ids

	def _reset_complete_callback(self):
		if self._start_task_timer is not None:
			self._start_task_timer.cancel()
			self._start_task_timer = None

		self.is_resetting = False
		self.task_running = False
		self.get_logger().info('Reset complete, attempting to start new task')

		self.start_new_task()

	def reset_nav2_non_blocking(self):
		"""Reset Nav2 without blocking the executor."""
		self.get_logger().info('Requesting costmap clears non-blocking...')

		if self.clear_global_costmap.wait_for_service(timeout_sec=0.1):
			req = ClearEntireCostmap.Request()
			future = self.clear_global_costmap.call_async(req)
			future.add_done_callback(lambda f: self.get_logger().info('Global costmap cleared'))
		else:
			self.get_logger().warning('Global costmap service not available')

		if self.clear_local_costmap.wait_for_service(timeout_sec=0.1):
			req = ClearEntireCostmap.Request()
			future = self.clear_local_costmap.call_async(req)
			future.add_done_callback(lambda f: self.get_logger().info('Local costmap cleared'))
		else:
			self.get_logger().warning('Local costmap service not available')

	def start_new_task_once(self):
		"""One-shot wrapper to call start_new_task and destroy the timer."""
		# This is a one-shot timer, so cancel it to ensure it doesn't execute again
		if hasattr(self, '_start_task_timer') and self._start_task_timer is not None:
				self._start_task_timer.cancel()
				self._start_task_timer = None

		# Now call the actual method
		self.get_logger().info('Timer calling start_new_task')
		self.start_new_task()

	def start_new_task(self):
		"""Start a new navigation task if not already navigating."""
		self.get_logger().info('Starting new navigation task')
		if self.task_running:
			self.get_logger().info('A task is already running')
			return

		if self.is_resetting:
			self.get_logger().info('Can\'t schedule a task, resetting')
			return

		if not self.nav_ready:
			self.get_logger().info('Nav not ready')
			return

		if not self.nav_client.wait_for_server(timeout_sec=0.1):
			self.get_logger().warning('Nav2 action server not responding, will retry later')
			# Schedule a retry
			self.create_timer(1.0, self.start_new_task_once, callback_group=self.callback_group)
			return

		self.get_logger().info('Sending random goal')
		new_goal = self.generate_random_goal()
		success = self.send_goal(new_goal)

		if success:
			self.task_running = True
		else:
			self.get_logger().error('Failed to send navigation goal')
			self.current_goal = None  # Reset current goal if send failed
			# Schedule a retry
			if not hasattr(self, 'is_resetting') or not self.is_resetting:
				self.create_timer(1.0, self.start_new_task_once, callback_group=self.callback_group)

	def reset_nav2(self):
		"""Gently reset Nav2 by clearing costmaps."""
		self.get_logger().info('Resetting Nav2 costmaps...')

		# Try to clear costmaps with both service naming patterns
		costmap_cleared = False

		# Try first pattern
		if self.clear_global_costmap.wait_for_service(timeout_sec=0.5):
			self.get_logger().info('Clearing global costmap')
			req = ClearEntireCostmap.Request()
			future = self.clear_global_costmap.call_async(req)
			rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
			costmap_cleared = True

		if self.clear_local_costmap.wait_for_service(timeout_sec=0.5):
			self.get_logger().info('Clearing local costmap')
			req = ClearEntireCostmap.Request()
			future = self.clear_local_costmap.call_async(req)
			rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
			costmap_cleared = True

		if not costmap_cleared:
			self.get_logger().warning('Could not clear costmaps - services not available')

		self.get_logger().info('Nav2 reset complete')

	@property
	def _robot_pos(self):
		return p.getBasePositionAndOrientation(self._robot_id)

	@property
	def robot_xy(self):
		pos, _ = self._robot_pos
		return pos[0], pos[1]

	def _apply_velocity(self):
		"""Apply velocity commands to the robot in PyBullet."""
		wheel_distance = 0.555  # Distance between left and right wheels (adjust for your robot)

		left_wheel_vel = (self.linear_vel - (wheel_distance / 2.0) * self.angular_vel) / 0.1651
		right_wheel_vel = (self.linear_vel + (wheel_distance / 2.0) * self.angular_vel) / 0.1651 # wheel radius

		#print(f'Applying {left_wheel_vel:.2f}, {right_wheel_vel:.2f}')

		wheel_indices = [2,3,4,5]  # Adjust these indices if needed
		max_force=100.

		for wheel in wheel_indices:
			if wheel in [2, 4]:  # Left wheels
				self._sim.setJointMotorControl2(
					bodyUniqueId=self._robot_id,
					jointIndex=wheel,
					controlMode=self._sim.VELOCITY_CONTROL,
					targetVelocity=left_wheel_vel,
					force=max_force
				)
			else:  # Right wheels
				self._sim.setJointMotorControl2(
					bodyUniqueId=self._robot_id,
					jointIndex=wheel,
					controlMode=self._sim.VELOCITY_CONTROL,
					targetVelocity=right_wheel_vel,
					force=max_force
				)

	def _cmd_vel_cb(self, msg):
		self.linear_vel = msg.linear.x
		self.angular_vel = msg.angular.z

	def _publish_camera_info(self):
		width = 320
		height = 240
		fov = 60

		focal_length = (width / 2.0) / math.tan(math.radians(fov / 2.0))

		camera_info_msg = CameraInfo()
		camera_info_msg.header.stamp = self.get_time_msg()
		camera_info_msg.header.frame_id = self.camera_frame

		camera_info_msg.width = width
		camera_info_msg.height = height

		camera_info_msg.distortion_model = "plumb_bob"
		camera_info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # No distortion

		camera_info_msg.k = [
			focal_length, 0.0, width / 2.0,
			0.0, focal_length, height / 2.0,
			0.0, 0.0, 1.0
		]

		camera_info_msg.r = [
			1.0, 0.0, 0.0,
			0.0, 1.0, 0.0,
			0.0, 0.0, 1.0
		]
		camera_info_msg.p = [
			focal_length, 0.0, width / 2.0, 0.0,
			0.0, focal_length, height / 2.0, 0.0,
			0.0, 0.0, 1.0, 0.0
		]

		# Publish the message
		self._camera_info_publisher.publish(camera_info_msg)

	def get_time_msg(self):
		time = Time()
		sim_sec = int(self.sim_time)
		sim_nano = int((self.sim_time - sim_sec) * 1e9)
		time.sec = sim_sec
		time.nanosec = sim_nano

		return time

	def _publish_static_transforms(self):
		base_link_to_footprint = TransformStamped()
		base_link_to_footprint.header.stamp = self.get_time_msg()
		base_link_to_footprint.header.frame_id = self.base_frame  # base_link
		base_link_to_footprint.child_frame_id = 'base_footprint'

		base_link_to_footprint.transform.translation.x = 0.0
		base_link_to_footprint.transform.translation.y = 0.0
		base_link_to_footprint.transform.translation.z = 0.0
		base_link_to_footprint.transform.rotation.x = 0.0
		base_link_to_footprint.transform.rotation.y = 0.0
		base_link_to_footprint.transform.rotation.z = 0.0
		base_link_to_footprint.transform.rotation.w = 1.0

		base_to_lidar = TransformStamped()
		base_to_lidar.header.stamp = self.get_time_msg()
		base_to_lidar.header.frame_id = self.base_frame
		base_to_lidar.child_frame_id = self.lidar_frame

		base_to_lidar.transform.translation.x = 0.0
		base_to_lidar.transform.translation.y = 0.0
		base_to_lidar.transform.translation.z = 0.5

		base_to_lidar.transform.rotation.x = 0.0
		base_to_lidar.transform.rotation.y = 0.0
		base_to_lidar.transform.rotation.z = 0.0
		base_to_lidar.transform.rotation.w = 1.0

		base_to_camera = TransformStamped()
		base_to_camera.header.stamp = self.get_time_msg()
		base_to_camera.header.frame_id = self.base_frame
		base_to_camera.child_frame_id = self.camera_frame

		base_to_camera.transform.translation.x = 0.0
		base_to_camera.transform.translation.y = 0.0
		base_to_camera.transform.translation.z = 0.5

		base_to_camera.transform.rotation.x = 0.0
		base_to_camera.transform.rotation.y = 0.0
		base_to_camera.transform.rotation.z = 0.0
		base_to_camera.transform.rotation.w = 1.0

		# Publish static transforms
		self.static_tf_broadcaster.sendTransform([base_link_to_footprint, base_to_lidar, base_to_camera])
		self.get_logger().info('Published static transforms')

	def _publish_lidar(self):
		t = self.get_time_msg()
		rays = 600
		lidar_position, lidar_orient = self._robot_pos
		lidar_position = np.array(lidar_position) + np.array([0, 0, 0.5])

		ranges = get_lidar_data(
			self._sim,
			lidar_position,
			lidar_orient,
			num_rays=rays
		)

		msg = LaserScan(
			header=Header(
				stamp=t,
				frame_id=self.lidar_frame
			),
			angle_min=0.,
			angle_max=2*math.pi,
			angle_increment=(2*math.pi / rays),
			time_increment=0.,
			scan_time=0.1,
			range_min=0.1,
			range_max=15.,
			ranges=[r.distance for r in ranges],
			#intensities=[1.0 for _ in ranges]
		)

		self._lidar_publisher.publish(msg)

	def _publish_camera(self):
		position, orientation = self._robot_pos
		img_arry = get_camera_data(self._sim, position, orientation)

		img_msg = Image()
		img_msg.header.stamp = self.get_time_msg()
		img_msg.header.frame_id = self.camera_frame
		img_msg.height = 240
		img_msg.width = 320
		img_msg.encoding = "rgb8"
		img_msg.is_bigendian = False
		img_msg.step = 3 * 320  # 3 bytes per pixel
		img_msg.data = img_arry.tobytes()

		self._camera_publisher.publish(img_msg)

	def _debug_visualize_lidar(self):
		"""Visualize lidar rays directly in PyBullet for debugging"""
		lidar_position, _ = self._robot_pos
		lidar_position = np.array(lidar_position) + np.array([0, 0, 0.5])  # Adjust to lidar height

		# Clear any previous debug lines
		# (You might need to store line IDs if this approach doesn't work)
		self._sim.removeAllUserDebugItems()

		# Get the lidar data
		ranges = get_lidar_data(
			self._sim,
			lidar_position,
			num_rays=30 # Use fewer rays for visualization clarity
		)

		# Draw each ray
		for r in ranges:
			angle = r.angle
			distance = r.distance

			# Calculate endpoint
			end_x = lidar_position[0] + distance * math.cos(angle)
			end_y = lidar_position[1] + distance * math.sin(angle)
			end_z = lidar_position[2]

			# Draw line for the ray (red for hits, blue for max distance)
			color = [1, 0, 0] if distance < 12.0 else [0, 0, 1]
			self._sim.addUserDebugLine(
				lidar_position,
				[end_x, end_y, end_z],
				color,
				lineWidth=1,
				lifeTime=0.1
			)

	def _publish_odom(self):
		position, orientation = self._robot_pos
		#linear, angular = p.getBaseVelocity(self._robot_id)
		world_linear, world_angular = p.getBaseVelocity(self._robot_id)

		odom_msg = Odometry()
		odom_msg.header.stamp = self.get_time_msg()
		odom_msg.header.frame_id = "odom"
		odom_msg.child_frame_id = self.base_frame

		# Set position and orientation
		odom_msg.pose.pose.position.x = position[0]
		odom_msg.pose.pose.position.y = position[1]
		odom_msg.pose.pose.position.z = position[2]

		# ...orientation...
		odom_msg.pose.pose.orientation.x = orientation[0]
		odom_msg.pose.pose.orientation.y = orientation[1]
		odom_msg.pose.pose.orientation.z = orientation[2]
		odom_msg.pose.pose.orientation.w = orientation[3]

		# ...covs...
		odom_msg.pose.covariance = [
			0.01, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.01, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.01, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.01, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.01, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.01
		]
		odom_msg.twist.covariance = [
			0.01, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.01, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.01, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.01, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.01, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.01
		]

		euler = p.getEulerFromQuaternion(orientation)
		yaw = euler[2]

		base_linear_x = world_linear[0] * math.cos(yaw) + world_linear[1] * math.sin(yaw)
		base_linear_y = -world_linear[0] * math.sin(yaw) + world_linear[1] * math.cos(yaw)


		# Set velocities
		odom_msg.twist.twist.linear.x = base_linear_x
		odom_msg.twist.twist.linear.y = base_linear_y
		odom_msg.twist.twist.linear.z = world_linear[2]
		odom_msg.twist.twist.angular.x = world_angular[0]
		odom_msg.twist.twist.angular.y = world_angular[1]
		odom_msg.twist.twist.angular.z = world_angular[2]

		# Transform stuff
		transform = TransformStamped()
		transform.header.stamp = self.get_time_msg()
		transform.header.frame_id = "odom"
		transform.child_frame_id = self.base_frame

		transform.transform.translation.x = position[0]
		transform.transform.translation.y = position[1]
		transform.transform.translation.z = position[2]

		transform.transform.rotation.x = orientation[0]
		transform.transform.rotation.y = orientation[1]
		transform.transform.rotation.z = orientation[2]
		transform.transform.rotation.w = orientation[3]

		self._odo_publisher.publish(odom_msg)
		self.tf_broadcaster.sendTransform(transform)

	def _step_simulation(self):
		for _ in range(1):
			p.stepSimulation()

	def sim_cb(self):
		if self.is_resetting:
			return

		self.sim_time += self.timestep
		clock_msg = Clock()
		time_msg = self.get_time_msg()
		clock_msg.clock = time_msg

		#self._clock_publisher.publish(clock_msg)

		#self._apply_velocity()

		self._step_simulation()

		self._publish_lidar()
		self._publish_camera_info()
		self._publish_camera()
		#self._publish_odom()
		self.publish_goal_marker()

		if not self.is_resetting:
			self.check_goal_status_and_maybe_update()

def main(args=None):
	rclpy.init(args=args)
	node = SimNode()

	# Use a multithreaded executor to prevent blocking
	executor = MultiThreadedExecutor()
	executor.add_node(node)

	try:
		executor.spin()
	except KeyboardInterrupt:
		pass
	finally:
		executor.shutdown()
		node.destroy_node()
		rclpy.shutdown()

if __name__ == '__main__':
	main()
