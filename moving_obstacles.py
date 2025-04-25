import numpy as np
import random
import math

def create_moving_obstacle(p, position, size=0.2, color=[0, 0.8, 0, 1], mass=0):
	# Ensure position has 3 elements
	if len(position) == 2:
		position = [position[0], position[1], size/2]

	# Create collision shape
	collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size/2, size/2, size/2])

	# Create visual shape with color
	visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[size/2, size/2, size/2], rgbaColor=color)

	# Create the body (use mass=0 for kinematic control)
	obstacle_id = p.createMultiBody(
		baseMass=mass,
		baseCollisionShapeIndex=collision_shape_id,
		baseVisualShapeIndex=visual_shape_id,
		basePosition=position
	)

	print(f"Created moving obstacle at position {position}, ID: {obstacle_id}")
	return obstacle_id

class MovingObstacle:
	"""Class to manage a moving obstacle with a predefined path"""
	def __init__(self, p, obstacle_id, path_type="circular", 
				 center=None, radius=2.0, speed=0.5, 
				 start_pos=None, end_pos=None, 
				 bounds=None):
		self.p = p
		self.obstacle_id = obstacle_id
		self.path_type = path_type
		self.speed = speed
		self.time = 0

		# Get initial position
		pos, _ = p.getBasePositionAndOrientation(obstacle_id)
		self.position = list(pos)

		# Set up path parameters
		if path_type == "circular":
			self.center = center if center is not None else [0, 0]
			self.radius = radius
			# Find initial angle
			dx = pos[0] - self.center[0]
			dy = pos[1] - self.center[1]
			self.angle = math.atan2(dy, dx)

		elif path_type == "linear":
			self.start_pos = start_pos if start_pos is not None else [pos[0], pos[1]]
			self.end_pos = end_pos if end_pos is not None else [pos[0] + 5, pos[1]]
			self.direction = 1  # 1 = forward, -1 = backward

		elif path_type == "random":
			self.bounds = bounds if bounds is not None else [-5, 5, -5, 5]
			self.target = [
				random.uniform(self.bounds[0], self.bounds[1]),
				random.uniform(self.bounds[2], self.bounds[3])
			]
			self.time_to_new_target = random.uniform(3, 8)

	def update(self, dt):
		self.time += dt

		if self.path_type == "circular":
			# Update angle based on speed and radius
			angular_vel = self.speed / self.radius
			self.angle += angular_vel * dt

			# Calculate new position
			new_x = self.center[0] + self.radius * math.cos(self.angle)
			new_y = self.center[1] + self.radius * math.sin(self.angle)
			self.position[0] = new_x
			self.position[1] = new_y

		elif self.path_type == "linear":
			# Calculate direction vector
			dx = self.end_pos[0] - self.start_pos[0]
			dy = self.end_pos[1] - self.start_pos[1]
			distance = math.sqrt(dx*dx + dy*dy)

			# Normalize and scale by speed and direction
			if distance > 0:
				dx = dx / distance * self.speed * dt * self.direction
				dy = dy / distance * self.speed * dt * self.direction

			# Update position
			self.position[0] += dx
			self.position[1] += dy

			# Check if we've reached an endpoint
			current_to_end = math.sqrt(
				(self.position[0] - self.end_pos[0])**2 + 
				(self.position[1] - self.end_pos[1])**2
			)
			current_to_start = math.sqrt(
				(self.position[0] - self.start_pos[0])**2 + 
				(self.position[1] - self.start_pos[1])**2
			)

			if (self.direction == 1 and current_to_end < self.speed * dt) or \
			   (self.direction == -1 and current_to_start < self.speed * dt):
				# Reverse direction
				self.direction *= -1

		elif self.path_type == "random":
			# Check if we need a new target
			dist_to_target = math.sqrt(
				(self.position[0] - self.target[0])**2 + 
				(self.position[1] - self.target[1])**2
			)

			if dist_to_target < self.speed * dt or self.time > self.time_to_new_target:
				# Generate new target
				self.target = [
					random.uniform(self.bounds[0], self.bounds[1]),
					random.uniform(self.bounds[2], self.bounds[3])
				]
				self.time = 0
				self.time_to_new_target = random.uniform(3, 8)

			# Move toward target
			if dist_to_target > 0:
				dx = (self.target[0] - self.position[0]) / dist_to_target * self.speed * dt
				dy = (self.target[1] - self.position[1]) / dist_to_target * self.speed * dt
				self.position[0] += dx
				self.position[1] += dy

		# Apply the new position in PyBullet
		self.p.resetBasePositionAndOrientation(
			self.obstacle_id,
			self.position,
			self.p.getBasePositionAndOrientation(self.obstacle_id)[1]  # Keep current orientation
		)

def add_moving_obstacles(
		p,
		num_obstacles=3,
		bounds=[-7, 7, -7, 7],
		min_size=0.3,
		max_size=1.0,
		min_speed=0.2,
		max_speed=1.0,
		min_distance_from_robot=2.0,
		robot_id=None
):
	moving_obstacles = []
	path_types = ["circular", "linear", "random"]
	colors = [
		[0, 0.8, 0, 1],  # Green
		[0, 0, 0.8, 1],  # Blue
		[0.8, 0, 0.8, 1], # Purple
		[0, 0.8, 0.8, 1], # Cyan
		[0.8, 0.5, 0, 1]  # Orange
	]

	# Get robot position if provided
	robot_pos = None
	if robot_id is not None:
		robot_pos, _ = p.getBasePositionAndOrientation(robot_id)
		robot_pos = robot_pos[:2]  # Only x, y

	for i in range(num_obstacles):
		# Generate random parameters
		size = random.uniform(min_size, max_size)
		speed = random.uniform(min_speed, max_speed)
		path_type = random.choice(path_types)
		color = colors[i % len(colors)]

		# Generate position
		valid_position = False
		attempts = 0
		position = None

		while not valid_position and attempts < 20:
			x = random.uniform(bounds[0], bounds[1])
			y = random.uniform(bounds[2], bounds[3])

			# Check distance from robot
			if robot_pos is not None:
				dist_to_robot = math.sqrt((x - robot_pos[0])**2 + (y - robot_pos[1])**2)
				if dist_to_robot < min_distance_from_robot:
					attempts += 1
					continue

			# Check distance from other obstacles
			too_close = False
			for obs in moving_obstacles:
				dist = math.sqrt((x - obs.position[0])**2 + (y - obs.position[1])**2)
				if dist < size + 1.0:  # Keep obstacles separated
					too_close = True
					break

			if not too_close:
				valid_position = True
				position = [x, y, size/2]  # Place on ground

		if not valid_position:
			print(f"Could not find valid position for obstacle {i+1} after 20 attempts")
			continue

		# Create obstacle in PyBullet
		obstacle_id = create_moving_obstacle(p, position, size, color)

		# Create path parameters based on path type
		if path_type == "circular":
			# Use random center point
			center = [
				random.uniform(bounds[0] + 3, bounds[1] - 3),
				random.uniform(bounds[2] + 3, bounds[3] - 3)
			]
			# Make radius based on position and bounds
			max_radius = min(
				abs(bounds[0] - center[0]), 
				abs(bounds[1] - center[0]),
				abs(bounds[2] - center[1]),
				abs(bounds[3] - center[1])
			) - 1.0
			radius = min(max_radius, random.uniform(1.5, 4.0))

			# Adjust initial position to be on the circle
			angle = random.uniform(0, 2 * math.pi)
			position[0] = center[0] + radius * math.cos(angle)
			position[1] = center[1] + radius * math.sin(angle)
			p.resetBasePositionAndOrientation(obstacle_id, position, [0, 0, 0, 1])

			moving_obstacle = MovingObstacle(
				p, obstacle_id, "circular", 
				center=center, radius=radius, speed=speed
			)

		elif path_type == "linear":
			# Create linear path
			direction = random.uniform(0, 2 * math.pi)
			length = random.uniform(3.0, 8.0)
			end_pos = [
				position[0] + length * math.cos(direction),
				position[1] + length * math.sin(direction)
			]

			# Make sure end position is within bounds
			end_pos[0] = max(bounds[0] + 1, min(bounds[1] - 1, end_pos[0]))
			end_pos[1] = max(bounds[2] + 1, min(bounds[3] - 1, end_pos[1]))

			moving_obstacle = MovingObstacle(
				p, obstacle_id, "linear", 
				start_pos=position[:2], end_pos=end_pos, speed=speed
			)

		else:  # random path
			moving_obstacle = MovingObstacle(
				p, obstacle_id, "random", 
				bounds=bounds, speed=speed
			)

		moving_obstacles.append(moving_obstacle)

	return moving_obstacles
