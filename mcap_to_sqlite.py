from tqdm.notebook import tqdm
import numpy as np
import torch
from cv_bridge import CvBridge
import rosbag2_py
import rclpy
from rclpy import serialization
from sensor_msgs.msg import LaserScan, Image, CameraInfo
import matplotlib.pyplot as plt
import sqlite3
from tqdm import tqdm
import numpy as np
from io import BytesIO
import cv2
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
import rosbag2_py
import sensor_msgs.msg
import nav_msgs.msg
import os

bridge = CvBridge()

storage_options = rosbag2_py.StorageOptions(
    uri='sim_data',
    storage_id="mcap"
)
converter_options = rosbag2_py.ConverterOptions(
    input_serialization_format="cdr",
    output_serialization_format="cdr"
)
topics = ['/camera', '/scan', '/ground_truth_occupancy_grid', '/odom']
bag_filter = rosbag2_py.StorageFilter()
bag_filter.topics = topics

# Create reader instance
reader = rosbag2_py.SequentialReader()

reader.open(storage_options, converter_options)
reader.set_filter(bag_filter)

db_path="data.sqlite"

conn = sqlite3.Connection(db_path)
c = conn.cursor()

c.execute('''
	create table if not exists images (
		id integer primary key,
		timestamp integer,
		image blob
	)
''')
c.execute('''
	create table if not exists scans (
		id integer primary key,
		timestamp integer,
		ranges blob
	)
''')
c.execute('''
	create table if not exists occupancy_grids (
		id integer primary key,
		timestamp integer,
		data blob
	)
''')
c.execute('''
	create table if not exists poses (
		id integer primary key,
		timestamp integer,
		x float,
		y float,
		theta float
	)
''')

def ros_to_img(img):
	deser = serialization.deserialize_message(img, message_type=Image)

	img_data = np.moveaxis((np.array(
		bridge.imgmsg_to_cv2(
			deser,
			desired_encoding='passthrough'
		)
	)), 2, 0)

	timestamp = deser.header.stamp.sec * 1e9 + deser.header.stamp.nanosec

	return timestamp, img_data.tobytes()

def ros_to_gt(img):
	deser = serialization.deserialize_message(img, message_type=Image)
	gt = np.array(bridge.imgmsg_to_cv2(deser, desired_encoding='passthrough'))
	timestamp = deser.header.stamp.sec * 1e9 + deser.header.stamp.nanosec

	return timestamp, gt.tobytes()

def ros_to_odom(odom):
	data = serialization.deserialize_message(odom, message_type=Odometry)
	timestamp = data.header.stamp.sec * 1e9 + data.header.stamp.nanosec

	x = data.pose.pose.position.x
	y = data.pose.pose.position.y
	theta = data.pose.pose.orientation.z

	return timestamp, x, y, theta

def scan_to_numpy(scan):
	data = serialization.deserialize_message(scan, message_type=LaserScan)
	ranges = np.array(data.ranges)
	timestamp = data.header.stamp.sec * 1e9 + data.header.stamp.nanosec

	return timestamp, ranges.tobytes()

def commit_pose_batch(batch):
	c = conn.cursor()
	c.executemany(
		"""
		insert into poses(
			timestamp,
			x,
			y,
			theta
		) values (
			?, ?, ?, ?
		)
		""",
		[
			ros_to_odom(b)
			for b
			in batch
		]
	)
	conn.commit()

def commit_img_batch(batch):
	c = conn.cursor()
	c.executemany(
		"""
		insert into images (
			timestamp,
			image
		) values (
			?, ?
		)
		""",
		[
			ros_to_img(b)
			for b
			in batch
		]
	)
	conn.commit()

def commit_gt_batch(batch):
	c = conn.cursor()
	c.executemany(
		"""
		insert into occupancy_grids (
			timestamp,
			data
		) values (
			?, ?
		)
		""",
		[
			ros_to_gt(b)
			for b
			in batch
		]
	)
	conn.commit()

def commit_lidar_batch(batch):
	c = conn.cursor()
	c.executemany(
		"""
		insert into scans (
			timestamp,
			ranges
		) values (
			?, ?
		)
		""",
		[
			scan_to_numpy(b)
			for b
			in batch
		]
	)
	conn.commit()

batch_size = 100
progress = tqdm(total=40000)
count = 0
imgs = []
scans = []
grids = []
poses = []

while reader.has_next():
	cur = reader.read_next()
	#print(f'Handling message from {cur[0]}')
	if cur[0] == '/scan':
		scans.append(cur[1])
		if len(scans) >= batch_size:
			commit_lidar_batch(scans)
			scans = []
	elif cur[0] == '/ground_truth_occupancy_grid':
		grids.append(cur[1])
		if len(grids) >= batch_size:
			commit_gt_batch(grids)
			grids = []
	elif cur[0] == '/camera':
		imgs.append(cur[1])
		if len(imgs) >= batch_size:
			commit_img_batch(imgs)
			imgs = []
	elif cur[0] == '/odom':
		poses.append(cur[1])
		if len(poses) >= batch_size:
			commit_pose_batch(poses)
			poses = []

	else:
		print(cur[0])

	progress.update()
	count += 1
