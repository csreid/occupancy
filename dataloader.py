import time
import torch
from cache import LRUCache
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import sqlite3
import numpy as np

def _row_factory(cur, row):
	gid = row[0]
	imdata = row[1]
	scandata = row[2]
	x = row[3]
	y = row[4]
	theta = row[5]
	griddata = row[6]

	img = (np.frombuffer(imdata, dtype=np.int8).reshape(3, 240, 320) / 255.).astype(np.float32)
	scan = np.frombuffer(scandata, dtype=np.float32)
	grid = np.frombuffer(griddata, dtype=np.float32).reshape(200, 200)

	return (gid, img, scan, grid, x, y, np.cos(theta), np.sin(theta))

class OccupancyDataLoader:
	def __init__(self, cv_fraction=0.):
		self.db_path = 'data.sqlite'
		self.conn = sqlite3.Connection(self.db_path)
		self.cache = LRUCache(max_size=4)

		c = self.conn.cursor()
		ep_query = """
			select
				count(*)
			from (
				select distinct group_id from data_points_view
			)
		"""
		c.execute(ep_query)
		n_episodes_total = c.fetchall()[0][0]
		self.conn.commit()

		self.n_cv_episodes = int(cv_fraction * n_episodes_total)
		self.n_episodes = n_episodes_total - self.n_cv_episodes
		all_episode_ids = [eid for eid in range(1, n_episodes_total)]

		self.episode_ids = all_episode_ids[:self.n_episodes]
		self.cv_episode_ids = all_episode_ids[self.n_episodes:]

		self.conn.row_factory = _row_factory

	def sample(self, n, for_cv=False):
		if not for_cv:
			episodes = np.random.choice(
				self.episode_ids,
				size=n
			)
		else:
			episodes = np.random.choice(
				self.cv_episode_ids,
				size=n
			)

		gid_map = {
			ep: {
				"imgs": [],
				"scans": [],
				"grids": [],
				"xs": [],
				"ys": [],
				"sins": [],
				"coss": []
			}
			for ep
			in episodes
		}

		to_fetch = []
		for ep in episodes:
			cached_data = self.cache.get(ep)
			if cached_data is not None:
				gid_map[ep] = cached_data
			else:
				to_fetch.append(ep)

		## Get what's left from the database

		if len(to_fetch) > 0:
			c = self.conn.cursor()
			#illegal_evil_qstring = ','.join('?'*len(to_fetch))
			illegal_evil_qstring = ','.join([str(i) for i in to_fetch])
			q = f"""
				select
					group_id,
					image,
					ranges,
					x,
					y,
					theta,
					data
				from data_points_view
				where group_id in ({illegal_evil_qstring})
				order by timestamp
			"""

			self.conn.commit()
			c.execute(q)
			results = c.fetchall()

			fetched_gid_map = {
				ep: {
					"imgs": [],
					"scans": [],
					"grids": [],
					"xs": [],
					"ys": [],
					"sins": [],
					"coss": []
				}
				for ep
				in to_fetch
			}

			for gid, img, scan, grid, x, y, s, c in results:
				fetched_gid_map[gid]["imgs"].append(img)
				fetched_gid_map[gid]["scans"].append(scan)
				fetched_gid_map[gid]["grids"].append(grid)
				fetched_gid_map[gid]["xs"].append(x)
				fetched_gid_map[gid]["ys"].append(y)
				fetched_gid_map[gid]["sins"].append(s)
				fetched_gid_map[gid]["coss"].append(c)

			gid_map.update(fetched_gid_map)

			for ep, data in fetched_gid_map.items():
				self.cache.put(ep, data)

		### This part is normal again

		img_batch = []
		scan_batch = []
		grid_batch = []
		pose_batch = []

		for ep in episodes:
			img_tens = torch.tensor(np.array(gid_map[ep]['imgs']))
			scan_tens = torch.tensor(np.array(gid_map[ep]['scans']))
			# Add noise to the lidar scan of +/- 5cm
			scan_tens = ((scan_tens * 12.) + torch.randn_like(scan_tens) * 0.05) / 12.
			grid_tens = torch.tensor(np.array(gid_map[ep]['grids']))

			x_tens = torch.tensor(np.array(gid_map[ep]['xs']))
			y_tens = torch.tensor(np.array(gid_map[ep]['ys']))
			sins_tens = torch.tensor(np.array(gid_map[ep]['sins']))
			coss_tens = torch.tensor(np.array(gid_map[ep]['coss']))

			pose_tens =  torch.stack([x_tens, y_tens, sins_tens, coss_tens], dim=1)

			img_batch.append(img_tens)
			scan_batch.append(scan_tens)
			grid_batch.append(grid_tens)
			pose_batch.append(pose_tens)

		imgs = pad_sequence(img_batch)
		scans = pad_sequence(scan_batch)
		grids = pad_sequence(grid_batch)
		poses = pad_sequence(pose_batch).float()

		initial_pose = poses[0]

		# Inputs are all the data up until the last recorded measurement
		X = (
			imgs[:-1],
			scans[:-1],
			initial_pose
		)

		# Optimization goals are occupancy grids and poses,
		# offset by 1 into the future
		Y = (grids[1:], poses[1:])

		return X, Y

if __name__ == '__main__':
	odl = OccupancyDataLoader()
	sample = odl.sample(8)
