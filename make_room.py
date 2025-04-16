def create_room(p, size=10.0, wall_height=2.0, wall_thickness=0.1):
	"""Create a square room with the specified dimensions."""
	half_size = size / 2.0

	# Create box collision shapes for walls
	wall_ids = []

	# Wall colors
	wall_color = [0.9, 0.9, 0.9, 1]  # Light gray

	# Create four walls
	# Wall 1: +X side
	wall1_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness/2, half_size, wall_height/2])
	wall1_body = p.createMultiBody(
		baseMass=0,  # Static object
		baseCollisionShapeIndex=wall1_id,
		basePosition=[half_size, 0, wall_height/2],
		baseOrientation=[0, 0, 0, 1]
	)
	p.changeVisualShape(wall1_body, -1, rgbaColor=wall_color)
	wall_ids.append(wall1_body)

	# Wall 2: -X side
	wall2_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness/2, half_size, wall_height/2])
	wall2_body = p.createMultiBody(
		baseMass=0,
		baseCollisionShapeIndex=wall2_id,
		basePosition=[-half_size, 0, wall_height/2],
		baseOrientation=[0, 0, 0, 1]
	)
	p.changeVisualShape(wall2_body, -1, rgbaColor=wall_color)
	wall_ids.append(wall2_body)

	# Wall 3: +Y side
	wall3_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_size, wall_thickness/2, wall_height/2])
	wall3_body = p.createMultiBody(
		baseMass=0,
		baseCollisionShapeIndex=wall3_id,
		basePosition=[0, half_size, wall_height/2],
		baseOrientation=[0, 0, 0, 1]
	)
	p.changeVisualShape(wall3_body, -1, rgbaColor=wall_color)
	wall_ids.append(wall3_body)

	# Wall 4: -Y side
	wall4_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_size, wall_thickness/2, wall_height/2])
	wall4_body = p.createMultiBody(
		baseMass=0,
		baseCollisionShapeIndex=wall4_id,
		basePosition=[0, -half_size, wall_height/2],
		baseOrientation=[0, 0, 0, 1]
	)
	p.changeVisualShape(wall4_body, -1, rgbaColor=wall_color)
	wall_ids.append(wall4_body)

	return wall_ids
