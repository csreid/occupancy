# ros path setup
import sys
import os
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import random

# Add ROS2 paths to Python path
ros_paths = [
	"/opt/ros/jazzy/lib/python3.12/site-packages",
]

for path in ros_paths:
	if path not in sys.path:
		sys.path.append(path)

from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan, Image, CameraInfo
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from rosgraph_msgs.msg import Clock
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Time
from std_msgs.msg import Header,  ColorRGBA
#from marti_common.msg import StringStamped
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
from moving_obstacles import add_moving_obstacles
from uuid import uuid4
from tqdm import tqdm

def create_occupancy_grid(p, obstacle_ids, grid_size=20.0, resolution=0.1):
	"""
	Create a ground-truth occupancy grid for the simulation environment.

	Args:
		p: PyBullet physics client
		obstacle_ids: List of IDs for all obstacles in the environment
		grid_size: Size of the environment in meters (assumed square)
		resolution: Resolution of the occupancy grid in meters/cell

	Returns:
		grid: 2D numpy array where 1=occupied, 0=free space
		grid_origin: (x, y) coordinates of the grid origin in world frame
	"""

	# Calculate grid dimensions based on resolution
	grid_cells = int(grid_size / resolution)
	grid = np.zeros((grid_cells, grid_cells), dtype=np.uint8)

	# The grid covers [-grid_size/2, grid_size/2] in both x and y
	grid_origin = (-grid_size/2, -grid_size/2)

	# Function to convert world coordinates to grid indices
	def world_to_grid(x, y):
		grid_x = int((x - grid_origin[0]) / resolution)
		grid_y = int((y - grid_origin[1]) / resolution)
		# Ensure indices are within grid bounds
		grid_x = max(0, min(grid_x, grid_cells-1))
		grid_y = max(0, min(grid_y, grid_cells-1))
		return grid_x, grid_y

	# Mark all cells occupied by obstacles
	for obs_id in obstacle_ids:
		# For complex shapes, we need to get their AABB (Axis-Aligned Bounding Box)
		aabb_min, aabb_max = p.getAABB(obs_id)

		# Get more detailed collision information using rays
		# Sample points within the AABB
		x_range = np.arange(aabb_min[0], aabb_max[0], resolution)
		y_range = np.arange(aabb_min[1], aabb_max[1], resolution)

		for x in x_range:
			for y in y_range:
				# Check if this point is inside or close to the obstacle
				# Cast a ray from slightly above the point downward
				start = [x, y, aabb_max[2] + 0.1]
				end = [x, y, aabb_min[2] - 0.1]

				ray_results = p.rayTest(start, end)

				# If ray hits the obstacle, mark the cell as occupied
				if ray_results[0][0] == obs_id:
					grid_x, grid_y = world_to_grid(x, y)
					grid[grid_y, grid_x] = 1  # Mark as occupied

	return grid, grid_origin

def add_cube(p, position, size=0.2, color=[1, 0, 0, 1], mass=0):
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
		self.timestep = 0
		self.max_timesteps = 1000
		# Use a callback group to allow concurrent callbacks
		self.callback_group = ReentrantCallbackGroup()

		self.progress_bar = tqdm(total=100000)

		# Create action client for navigation
		self.nav_client = ActionClient(
			self,
			NavigateToPose,
			'navigate_to_pose',
			callback_group=self.callback_group
		)
		self._sim = get_sim()
		self.delta_t = 1/60.
		self.sim_time = 0.

		self.base_frame='base_link'
		self.lidar_frame='lidar_link'
		self.camera_frame='camera_link'

		#self.linear_vel = 0.
		#self.angular_vel = 0.

		self.tf_broadcaster = TransformBroadcaster(self)
		self.static_tf_broadcaster = StaticTransformBroadcaster(self)
		self.episode_id = uuid4()

		self._odo_publisher = self.create_publisher(
			Odometry,
			'odom',
			10
		)

#		self._episode_publisher = self.create_publisher(
#			StringStamped,
#			'episode_id',
#			10
#		)

		self._gt_publisher = self.create_publisher(
			Image,
			'ground_truth_occupancy_grid',
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

		## Stuff for managing the sim
		self._robot_id = None
		self.obstacle_ids = []
		self.x_bounds = (-7.0, 7.0)
		self.y_bounds = (-7.0, 7.0)

		self._timer = self.create_timer(0.01, self.sim_cb)
		self._clock_publisher = self.create_publisher(Clock, '/clock', 10)
		# Create timer for periodic status checking
		#self.timer = self.create_timer(1.0, self.check_nav2_status, callback_group=self.callback_group)

		self._start_task_timer = None
		self.task_running = False
		self.is_resetting = False

		self.reset_pybullet_simulation()

	def maybe_reset(self):
		if self.timestep >= self.max_timesteps:
			self.reset_pybullet_simulation()
			self.timestep = 0
			self.episode_id = uuid4()

	def _publish_episode(self):
		msg = StringStamped()
		msg.header.stamp = self.get_time_msg()
		msg.header.frame_id = 'base_link'

		msg.string = str(self.episode_id)

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

	def random_position(self):
		new_x = random.uniform(*self.x_bounds)
		new_y = random.uniform(*self.y_bounds)
		new_theta = random.uniform(-math.pi, math.pi)

		return new_x, new_y, new_theta

	def _publish_gt_grid(self):
		grid, _ = create_occupancy_grid(
			self._sim,
			self.obstacle_ids
		)

		img_msg = Image()
		img_msg.header.stamp = self.get_time_msg()
		img_msg.header.frame_id = 'base_link'
		img_msg.height = grid.shape[0]
		img_msg.width = grid.shape[1]

		img_msg.encoding = "32FC1"
		img_msg.is_bigendian = False
		img_msg.step = grid.shape[1] * 4

		grid = grid.astype(np.float32)
		img_msg.data = grid.tobytes()

		self._gt_publisher.publish(img_msg)

	def reset_pybullet_simulation(self):
		# Clear the simulation
		if self._robot_id is not None:
			self._sim.removeBody(self._robot_id)

		for obs_id in self.obstacle_ids:
			self._sim.removeBody(obs_id)

		self.obstacle_ids = []

		# Load the robot
		start_x, start_y, start_theta = self.random_position()
		#start_x, start_y, start_theta = (0, 0, 0)

		start_pos = [start_x, start_y, 0.1]
		#start_pos = [0, 0, 0.1]
		start_theta = 0.
		start_orientation = p.getQuaternionFromEuler([0, 0, start_theta])
		robot_id = p.loadURDF("husky/husky.urdf", start_pos, start_orientation)

		self._robot_id = robot_id

		# Load obstacles
		wall_ids = create_room(p, size=20)
		n_obstacles = np.random.randint(5, 10)

		for i in range(n_obstacles):
			size = np.random.uniform(1., 3.)
			obs_x, obs_y, obs_theta = self.random_position()

			# Give the robot a 1m circle of safety
			obs_dist = np.linalg.norm(
				np.array([obs_x, obs_y]) - np.array([start_x, start_y])
			)
			while (obs_dist - (size / 2)) < 1.:
				obs_x, obs_y, obs_theta = self.random_position()
				obs_dist = np.linalg.norm(
					np.array([obs_x, obs_y]) - np.array([start_x, start_y])
				)

			obs_id = add_cube(
				self._sim,
				position=[obs_x, obs_y, 0.5],
				size=size,
				mass=100
			)

			self.obstacle_ids.append(obs_id)

		self.obstacle_ids += wall_ids

		self.moving_obstacles = add_moving_obstacles(
			self._sim,
			num_obstacles=5,
			robot_id = self._robot_id
		)

		self.obstacle_ids += [mobs.obstacle_id for mobs in self.moving_obstacles]

		# Choose a random linear/angular velocity for the robot
		self.linear_vel = np.random.uniform(-1, 1)
		self.angular_vel = np.random.uniform(-1, 1)

		# Let things settle
		for _ in range(100):
			self._sim.stepSimulation()
			self.sim_time += self.delta_t

	@property
	def _robot_pos(self):
		return p.getBasePositionAndOrientation(self._robot_id)

	@property
	def robot_xy(self):
		pos, _ = self._robot_pos
		return pos[0], pos[1]

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
		for _ in range(10):
			p.stepSimulation()

	def _update_obstacles(self):
		dt = 1/6.

		for obs in self.moving_obstacles:
			obs.update(dt)

	def _apply_velocity(self):
		"""Apply velocity commands to the robot in PyBullet."""
		wheel_distance = 0.555

		left_wheel_vel = (self.linear_vel - (wheel_distance / 2.0) * self.angular_vel) / 0.1651
		right_wheel_vel = (self.linear_vel + (wheel_distance / 2.0) * self.angular_vel) / 0.1651 # wheel radius

		#print(f'Applying {left_wheel_vel:.2f}, {right_wheel_vel:.2f}')

		wheel_indices = [2,3,4,5]
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

	def sim_cb(self):
		if self.is_resetting:
			return

		self.timestep += 10
		self.sim_time += (self.delta_t * 10)
		clock_msg = Clock()
		time_msg = self.get_time_msg()
		clock_msg.clock = time_msg

		self._step_simulation()

		self._update_obstacles()
		self._publish_odom()
		self._publish_lidar()
		self._publish_camera_info()
		self._publish_camera()
		self._publish_gt_grid()
		self._publish_static_transforms()
		self._apply_velocity()
		#self._publish_episode()
		self.maybe_reset()

		self.progress_bar.update(10)

		if self.timestep >= 100000:
			sys.exit(0)

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
