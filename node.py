# ros path setup
import sys
import os

# Add ROS2 paths to Python path
ros_paths = [
	"/opt/ros/jazzy/lib/python3.12/site-packages",
	# Add any other ROS paths you need
]

for path in ros_paths:
	if path not in sys.path:
		sys.path.append(path)

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image, CameraInfo
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Header
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

def add_debug_cube(p, position, size=0.2, color=[1, 0, 0, 1], mass=0):
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
	p.setTimeStep(1/10.)

	# Load the ground plane
	planeId = p.loadURDF("plane.urdf")
	print(f"Ground plane loaded: {'Success' if planeId >= 0 else 'Failed'}")

	# Load the robot
	startPos = [0., 0., 0.1]
	startOrientation = p.getQuaternionFromEuler([0, 0, 0])
	robotId = p.loadURDF("husky/husky.urdf", startPos, startOrientation)

	wall_ids = create_room(p, size=20)

	# Create a directional light
	p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
	lightDirection = [0.52, 0.8, 0.7]
	p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 1)
	p.configureDebugVisualizer(p.COV_ENABLE_PLANAR_REFLECTION, 0)

	add_debug_cube(p, position=[2., 0., 0.3], size=1., mass=100)

	return p, robotId

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
		self._sim, self._robot_id = get_sim()
		self._iterations = 0

		self.base_frame='base_link'
		self.lidar_frame='lidar_link'
		self.camera_frame='camera_link'

		self.linear_vel = 0.
		self.angular_vel = 0.

		self.tf_broadcaster = TransformBroadcaster(self)
		self.static_tf_broadcaster = StaticTransformBroadcaster(self)
		self._publish_static_transforms()

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

		self._timer = self.create_timer(0.01, self.sim_cb)
		self._clock_publisher = self.create_publisher(Clock, '/clock', 10)

	@property
	def _robot_pos(self):
		return p.getBasePositionAndOrientation(self._robot_id)

	def _apply_velocity(self):
		"""Apply velocity commands to the robot in PyBullet."""
		wheel_distance = 0.555  # Distance between left and right wheels (adjust for your robot)

		left_wheel_vel = (self.linear_vel - (wheel_distance / 2.0) * self.angular_vel) / 0.1651
		right_wheel_vel = (self.linear_vel + (wheel_distance / 2.0) * self.angular_vel) / 0.1651 # wheel radius

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
		camera_info_msg.header.stamp = self.get_clock().now().to_msg()
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

	def _publish_static_transforms(self):
		base_link_to_footprint = TransformStamped()
		base_link_to_footprint.header.stamp = self.get_clock().now().to_msg()
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
		base_to_lidar.header.stamp = self.get_clock().now().to_msg()
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
		base_to_camera.header.stamp = self.get_clock().now().to_msg()
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
		t = self.get_clock().now().to_msg()
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
		img_msg.header.stamp = self.get_clock().now().to_msg()
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

	def _publish_transforms(self, timestamp):
		# During first few iterations, publish map→odom to help bootstrap
		if self._iterations < 50:  # Only during startup
			map_to_odom = TransformStamped()
			map_to_odom.header.stamp = timestamp
			map_to_odom.header.frame_id = 'map'
			map_to_odom.child_frame_id = 'odom'
			# Identity transform (0,0,0 position, no rotation)
			map_to_odom.transform.translation.x = 0.0
			map_to_odom.transform.translation.y = 0.0
			map_to_odom.transform.translation.z = 0.0
			map_to_odom.transform.rotation.x = 0.0
			map_to_odom.transform.rotation.y = 0.0
			map_to_odom.transform.rotation.z = 0.0
			map_to_odom.transform.rotation.w = 1.0
			
			self.tf_broadcaster.sendTransform(map_to_odom)
			self._iterations += 1

	def _publish_odom(self):
		position, orientation = self._robot_pos
		#linear, angular = p.getBaseVelocity(self._robot_id)
		world_linear, world_angular = p.getBaseVelocity(self._robot_id)

		odom_msg = Odometry()
		odom_msg.header.stamp = self.get_clock().now().to_msg()
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
		transform.header.stamp = self.get_clock().now().to_msg()
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
		for _ in range(2):
			p.stepSimulation()

	def sim_cb(self):
		clock_msg = Clock()
		sim_time = self.get_clock().now().to_msg()
		clock_msg.clock = sim_time
		self._clock_publisher.publish(clock_msg)

		self._apply_velocity()

		self._step_simulation()

		self._publish_lidar()
		self._publish_camera_info()
		self._publish_camera()
		self._publish_odom()

def main(args=None):
	rclpy.init(args=args)
	sim = SimNode()
	rclpy.spin(sim)

	sim.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
