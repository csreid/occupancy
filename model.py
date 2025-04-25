import torch
import math
from torch.nn import LSTM, Linear, Module, Conv2d, MaxPool2d, Conv1d, MaxPool1d, ConvTranspose2d, Upsample
import matplotlib.pyplot as plt

def _output_size(input_size, conv, mp):
	ks_w, ks_h = conv.kernel_size
	p_w, p_h = conv.padding
	s_w, s_h = conv.stride

	out_w = math.floor(((input_size[0] - ks_w + 2 * p_w) / s_w) + 1)
	out_h = math.floor(((input_size[1] - ks_h + 2 * p_h) / s_h) + 1)

	try:
		ks_w, ks_h = mp.kernel_size
	except:
		ks_w, ks_h = (mp.kernel_size, mp.kernel_size)

	try:
		p_w, p_h = mp.padding
	except:
		p_w, p_h = (mp.padding, mp.padding)

	try:
		s_w, s_h = mp.stride
	except:
		s_w, s_h = (mp.stride, mp.stride)

	out_w = math.floor(((out_w + 2 * p_w - 1) / s_w) + 1)
	out_h = math.floor(((out_h + 2 * p_h - 1) / s_h) + 1)

	return out_w, out_h

class VisionModule(Module):
	def __init__(self):
		super().__init__()

		self._img_width = 320
		self._img_height = 240

		self.cnn1 = Conv2d(
			in_channels=3,
			out_channels=16,
			kernel_size=4,
			stride=2,
		)

		self.mp1 = MaxPool2d(
			kernel_size=4,
			stride=4,
		)

		ow, oh = _output_size((self._img_width, self._img_height), self.cnn1, self.mp1)

		self.cnn2 = Conv2d(
			in_channels=16,
			out_channels=32,
			kernel_size=3,
			stride=1,
		)

		self.mp2 = MaxPool2d(
			kernel_size=3,
			stride=3,
		)

		ow, oh = _output_size((ow, oh,), self.cnn2, self.mp2)

		self.cnn3 = Conv2d(
			in_channels=32,
			out_channels=64,
			kernel_size=3,
			stride=1
		)
		self.mp3 = MaxPool2d(
			kernel_size=3,
			stride=3,
		)

		ow, oh = _output_size((ow, oh), self.cnn3, self.mp3)

		self.linear = Linear(384, 128)

	def compute_output_shape(self):
		test_in = torch.zeros((1, 3, self._img_height, self._img_width))
		out_shape = self.forward(test_in).shape

		return out_shape

	def forward(self, X):
		out = self.cnn1(X)
		out = self.mp1(out)

		out = self.cnn2(out)
		out = self.mp2(out)

		out = self.cnn3(out)
		out = self.mp3(out)

		out = torch.flatten(out, start_dim=1)
		out = self.linear(out)

		return out

class LidarModule(Module):
	def __init__(self):
		super().__init__()
		self.cnn1 = Conv1d(
			in_channels=1,
			out_channels=16,
			kernel_size=8,
			stride=2,
			padding_mode='circular'
		)
		self.mp1 = MaxPool1d(
			kernel_size=8
		)

		self.cnn2 = Conv1d(
			in_channels=16,
			out_channels=64,
			kernel_size=4,
			stride=2,
			padding_mode='circular'
		)
		self.mp2 = MaxPool1d(
			kernel_size=4
		)

		self.linear = Linear(64*4, 128)

	def forward(self, X):
		out = self.cnn1(X)
		out = self.mp1(out)

		out = self.cnn2(out)
		out = self.mp2(out)

		out = torch.flatten(out, start_dim=1)

		out = self.linear(out)

		return out

class RecurrentModule(Module):
	def __init__(self):
		super().__init__()
		self._input_size = 128*2 #128 for each of LIDAR and vision data
		self._rnn = LSTM(self._input_size, 512)

	def forward(self, X):
		return self._rnn(X)

class OutputModule(Module):
	def __init__(self):
		super().__init__()
		# Input to deconv should be 128 channels of 2x2
		self._deconv1 = ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=2)
		self._ups1 = Upsample(size=(6,6))

		self._deconv2 = ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=4, stride=2)
		self._ups2 = Upsample(size=(32, 32))

		self._deconv3 = ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=8, stride=4)
		self._ups3 = Upsample(size=(200, 200))

	def compute_output_shape(self):
		test_in = torch.zeros((1, 512))
		out_shape = self.forward(test_in).shape

		return out_shape

	def forward(self, X):
		# Input is sequence_length, batch_size, features
		in_shape = X.shape
		out = X.reshape((-1, 128, 2, 2))
		out = self._deconv1(out)
		out = self._ups1(out)

		out = self._deconv2(out)
		out = self._ups2(out)

		out = self._deconv3(out)
		out = self._ups3(out)

		# Reshape back to original
		out = out.reshape(in_shape[0], in_shape[1], 1, 200, 200)

		return out

class OccupancyGridModule(Module):
	def __init__(self):
		super().__init__()
		self._vision = VisionModule()
		self._lidar = LidarModule()
		self._rnn = RecurrentModule()
		self._output = OutputModule()

	def forward(self, lidar, camera):
		# Input shape for camera will be:
		#     sequence_length x batch_size x channels x height x width
		# Need to squash batch_size x sequence_length into one:
		camera_shape = camera.shape
		camera = camera.reshape(-1, camera_shape[2], camera_shape[3], camera_shape[4])

		# Similar for lidar, but input should be sequence_length x batch_size x 600 (# of rays):
		lidar_shape = lidar.shape
		lidar = lidar.reshape(-1, 1, 600) # <batch_size>, 1 channel, 600 features

		# And then reshape back to original
		vision_feats = self._vision(camera).reshape(camera_shape[0], camera_shape[1], -1)
		lidar_feats = self._lidar(lidar).reshape(lidar_shape[0], lidar_shape[1], -1)

		feats = torch.cat([vision_feats, lidar_feats], dim=2)

		out, _ = self._rnn(feats)
		out = self._output(out)

		return out

if __name__ == '__main__':
	vm = VisionModule()
	lm = LidarModule()
	om = OutputModule()
	og = OccupancyGridModule()

	test_imgs = torch.zeros((100, 16, 3, 240, 320))
	test_lidar = torch.zeros((100, 16, 600))

	og(lidar=test_lidar, camera=test_imgs)
