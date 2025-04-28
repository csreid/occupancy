import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from torch.utils.tensorboard import SummaryWriter
import io
from PIL import Image

class OdometryVisualizer:
	"""
	Helper class to visualize odometry data (ground truth vs. predicted) using TensorBoard.
	"""
	def __init__(self, writer):
		"""
		Initialize the visualizer with a TensorBoard log directory.

		Args:
			log_dir: Directory where TensorBoard logs will be saved
		"""
		self.writer = writer

	def plot_trajectory(self, target_odom, pred_odom, step, tag='odometry/trajectory'):
		"""
		Plot trajectory comparison between target and predicted odometry.

		Args:
			target_odom: Target odometry tensor of shape [seq_len, 3] (x, y, theta)
			pred_odom: Predicted odometry tensor of shape [seq_len, 3] (x, y, theta)
			step: Global step for TensorBoard logging
			tag: Tag for the plot in TensorBoard
		"""
		# Convert tensors to numpy for plotting
		if isinstance(target_odom, torch.Tensor):
			target_odom = target_odom.detach().cpu().numpy()
		if isinstance(pred_odom, torch.Tensor):
			pred_odom = pred_odom.detach().cpu().numpy()

		# Create figure
		fig = Figure(figsize=(10, 8))
		ax = fig.add_subplot(111)

		# Plot target trajectory
		ax.plot(target_odom[:, 0], target_odom[:, 1], 'b-', linewidth=2, label='Target')

		# Plot predicted trajectory
		ax.plot(pred_odom[:, 0], pred_odom[:, 1], 'r--', linewidth=2, label='Predicted')

		# Plot orientation arrows (every nth point)
		n = max(1, len(target_odom) // 10)  # Show max 10 arrows
		for i in range(0, len(target_odom), n):
			# Target orientation
			dx = 0.2 * np.cos(target_odom[i, 2])
			dy = 0.2 * np.sin(target_odom[i, 2])
			ax.arrow(target_odom[i, 0], target_odom[i, 1], dx, dy, 
					 head_width=0.05, head_length=0.1, fc='blue', ec='blue')

			# Predicted orientation
			dx = 0.2 * np.cos(pred_odom[i, 2])
			dy = 0.2 * np.sin(pred_odom[i, 2])
			ax.arrow(pred_odom[i, 0], pred_odom[i, 1], dx, dy,
					 head_width=0.05, head_length=0.1, fc='red', ec='red')

		# Mark start and end points
		ax.plot(target_odom[0, 0], target_odom[0, 1], 'bo', markersize=8, label='Start')
		ax.plot(target_odom[-1, 0], target_odom[-1, 1], 'go', markersize=8, label='End (Target)')
		ax.plot(pred_odom[-1, 0], pred_odom[-1, 1], 'ro', markersize=8, label='End (Predicted)')

		# Set labels and title
		ax.set_xlabel('X Position')
		ax.set_ylabel('Y Position')
		ax.set_title('Robot Trajectory: Target vs. Predicted')
		ax.legend()
		ax.grid(True)

		# Make axes equal for proper visualization
		ax.axis('equal')

		# Add the plot to TensorBoard
		self._figure_to_tensorboard(fig, tag, step)

	def _figure_to_tensorboard(self, figure, tag, step):
		"""Convert a matplotlib figure to a TensorBoard image."""
		# Save the plot to a PNG in memory
		buf = io.BytesIO()
		figure.savefig(buf, format='png', dpi=100)
		buf.seek(0)

		# Convert to PIL Image
		image = Image.open(buf)

		# Convert to numpy array
		image_np = np.array(image)

		# Add to TensorBoard
		self.writer.add_image(tag, image_np, step, dataformats='HWC')

	def close(self):
		"""Close the SummaryWriter."""
		self.writer.close()
