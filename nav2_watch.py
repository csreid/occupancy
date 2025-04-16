#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ClearEntireCostmap
from geometry_msgs.msg import PoseStamped
import random
import math
import time
from tf2_ros import Buffer, TransformListener
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

class Nav2Manager(Node):
	def __init__(self):
		super().__init__('nav2_manager')

		# Use a callback group to allow concurrent callbacks
		self.callback_group = ReentrantCallbackGroup()

		# Create action client for navigation
		self.nav_client = ActionClient(
			self, 
			NavigateToPose, 
			'navigate_to_pose', 
			callback_group=self.callback_group
		)

		# Service clients for resetting costmaps - handle both namespaces
		# Try the standard namespace first
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

		# Also try with leading slash
		self.clear_global_costmap_alt = self.create_client(
			ClearEntireCostmap, 
			'/global_costmap/clear_entirely_global_costmap',
			callback_group=self.callback_group
		)
		self.clear_local_costmap_alt = self.create_client(
			ClearEntireCostmap, 
			'/local_costmap/clear_entirely_local_costmap',
			callback_group=self.callback_group
		)

		# Setup TF listener to get robot position
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

		# Define the simulation bounds for random goal generation
		self.x_bounds = (-5.0, 5.0)  # Adjust based on your environment
		self.y_bounds = (-5.0, 5.0)  # Adjust based on your environment

		# Track current navigation goal
		self.current_goal = None
		self.goal_handle = None
		self.nav_ready = False

		# Create timer for periodic status checking
		self.timer = self.create_timer(1.0, self.check_nav2_status, callback_group=self.callback_group)

		self.get_logger().info('Nav2 Manager initialized')

	def check_nav2_status(self):
		"""Check if Nav2 action server is available."""
		if self.current_goal is not None:
			# Already navigating, nothing to do
			return

		# Check if Nav2 action server is available
		if self.nav_client.wait_for_server(timeout_sec=0.1):
			if not self.nav_ready:
				self.get_logger().info('Nav2 navigation action server is ready')
				self.nav_ready = True

			# If we have services for costmap clearing, we're really ready
			if (self.clear_global_costmap.wait_for_service(timeout_sec=0.1) or
				self.clear_global_costmap_alt.wait_for_service(timeout_sec=0.1)):
				self.start_new_task()
			else:
				self.get_logger().info('Waiting for costmap services...')
		else:
			self.nav_ready = False
			self.get_logger().info('Waiting for Nav2 action server...')

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

		# Try alternative pattern if first one failed
		if not costmap_cleared:
			if self.clear_global_costmap_alt.wait_for_service(timeout_sec=0.5):
				self.get_logger().info('Clearing global costmap (alt)')
				req = ClearEntireCostmap.Request()
				future = self.clear_global_costmap_alt.call_async(req)
				rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
				costmap_cleared = True

			if self.clear_local_costmap_alt.wait_for_service(timeout_sec=0.5):
				self.get_logger().info('Clearing local costmap (alt)')
				req = ClearEntireCostmap.Request()
				future = self.clear_local_costmap_alt.call_async(req)
				rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
				costmap_cleared = True

		if not costmap_cleared:
			self.get_logger().warning('Could not clear costmaps - services not available')

		self.get_logger().info('Nav2 reset complete')

	def get_robot_position(self):
		"""Get current robot position from TF if available."""
		try:
			transform = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
			x = transform.transform.translation.x
			y = transform.transform.translation.y

			# Get a point that's at least 3 meters away from current position
			while True:
				new_x = random.uniform(*self.x_bounds)
				new_y = random.uniform(*self.y_bounds)

				# Calculate distance to ensure it's far enough
				distance = math.sqrt((new_x - x)**2 + (new_y - y)**2)
				if distance >= 3.0:
					return new_x, new_y

		except Exception as e:
			self.get_logger().warning(f'Could not get robot position: {e}')
			# Fall back to fully random position
			return random.uniform(*self.x_bounds), random.uniform(*self.y_bounds)

	def generate_random_goal(self):
		"""Generate a random navigation goal within the defined bounds."""
		x, y = self.get_robot_position()
		theta = random.uniform(-math.pi, math.pi)

		goal = PoseStamped()
		goal.header.frame_id = 'map'
		goal.header.stamp = self.get_clock().now().to_msg()
		goal.pose.position.x = x
		goal.pose.position.y = y
		goal.pose.position.z = 0.0

		# Convert theta to quaternion (yaw only)
		goal.pose.orientation.x = 0.0
		goal.pose.orientation.y = 0.0
		goal.pose.orientation.z = math.sin(theta / 2.0)
		goal.pose.orientation.w = math.cos(theta / 2.0)

		self.get_logger().info(f'Generated random goal: x={x:.2f}, y={y:.2f}, theta={theta:.2f}')
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

	def goal_response_callback(self, future):
		"""Callback when goal is accepted or rejected."""
		goal_handle = future.result()

		if not goal_handle.accepted:
			self.get_logger().error('Goal was rejected')
			self.current_goal = None
			# Try again after a short delay
			self.create_timer(2.0, self.start_new_task, callback_group=self.callback_group)
			return

		self.get_logger().info('Goal accepted')
		self.goal_handle = goal_handle

		# Get result async
		result_future = goal_handle.get_result_async()
		result_future.add_done_callback(self.goal_result_callback)

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
			self.reset_simulation_and_nav()

		except Exception as e:
			self.get_logger().error(f'Error in goal result callback: {e}')
			self.current_goal = None
			self.goal_handle = None
			# Try again after a short delay
			self.create_timer(5.0, self.start_new_task, callback_group=self.callback_group)

	def reset_simulation_and_nav(self):
		"""Reset both the simulation and Nav2."""
		self.get_logger().info('Resetting simulation and navigation...')

		# Reset Nav2 components
		self.reset_nav2()

		# Here you would add code to reset your PyBullet simulation
		# This is where you would randomize the environment
		# Example (replace with your actual simulation reset code):
		# self.reset_pybullet_simulation()

		# Clear current goal tracking
		self.current_goal = None
		self.goal_handle = None

		# Wait a moment for systems to initialize after reset
		self.create_timer(2.0, self.start_new_task, callback_group=self.callback_group,)

	def start_new_task(self):
		"""Start a new navigation task if not already navigating."""
		if self.current_goal is not None:
			return

		if not self.nav_ready:
			return

		self.get_logger().info('Starting new navigation task')
		new_goal = self.generate_random_goal()
		self.send_goal(new_goal)

def main(args=None):
	rclpy.init(args=args)

	node = Nav2Manager()

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
