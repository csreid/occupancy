from tqdm import tqdm
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
import torch
from torch.optim import Adam
from dataloader import OccupancyDataLoader
from model import OccupancyGridModule
from torch.utils.tensorboard import SummaryWriter
from odom_vis import OdometryVisualizer

BATCH_SIZE=16
ITERATIONS=10000
STEP_ITERS=10

if torch.cuda.is_available():
	dev = 'cuda:0'
else:
	dev = 'cpu'

writer = SummaryWriter()
model = OccupancyGridModule().to(dev)
try:
	model.load_state_dict(torch.load('model.pt', weights_only=False))
except:
	print(f'Could not load a model, starting fresh')

opt = Adam(model.parameters(),) #weight_decay=0.01)
loader = OccupancyDataLoader(cv_fraction=0.1)

loss_fn_grid = BCEWithLogitsLoss()
loss_fn_odom = MSELoss()

ov = OdometryVisualizer(writer)

def validate(model):
	with torch.no_grad():
		(imgs, scans, init_pose), (grids, poses) = loader.sample(12, for_cv=True)
		pred, pred_pose = model(scans.to(dev), imgs.to(dev), init_pose.to(dev))
		loss_grid = loss_fn_grid(
				pred.squeeze().flatten(start_dim=0, end_dim=1),
				grids.flatten(start_dim=0, end_dim=1).to(dev),
		)
		loss_odom = loss_fn_odom(
			pred_pose,
			poses.to(dev)
		)
		return loss_grid, loss_odom

def log_sample(i):
	(sample_imgs, sample_scans, sample_init_pose), (grids, poses) = loader.sample(1, for_cv=False)
	with torch.no_grad():
		sample_out, pred_poses = model(
			sample_scans.to(dev),
			sample_imgs.to(dev),
			sample_init_pose.to(dev)
		)

		sample_out = torch.sigmoid(
			sample_out.squeeze().unsqueeze(0).unsqueeze(2).to(dev)
		)
		grids = grids.squeeze().unsqueeze(0).unsqueeze(2).to(dev)

		video = torch.cat((grids, sample_out), dim=-1).expand(-1, -1, 3, -1, -1) # Pop the one channel into RGB to treat it like a video

		writer.add_video('Sample/Grids', video, i)
		writer.add_video('Sample/Camera',  sample_imgs.swapaxes(0, 1), i)

		ov.plot_trajectory(target_odom=poses[:, 0], pred_odom=pred_poses[:, 0], step=i)

try:
	for i in tqdm(range(ITERATIONS)):
		(imgs, scans, init_pose), (grids, poses) = loader.sample(BATCH_SIZE)
		pred_grid, pred_pose = model(
			scans.to(dev),
			imgs.to(dev),
			init_pose.to(dev)
		)

		# To make this work, gotta squeeze seq_length
		# and batch size into the same dim
		loss_grid = loss_fn_grid(
				pred_grid.flatten(start_dim=0, end_dim=1).to(dev),
				grids.flatten(start_dim=0, end_dim=1).unsqueeze(1).to(dev),
		)

		loss_odom = loss_fn_odom(
			pred_pose,
			poses.to(dev)
		)
		writer.add_scalar('Loss/Occupancy Grid Loss', loss_grid, i)
		writer.add_scalar('Loss/Odom Loss', loss_odom, i)

		loss = 20*loss_grid + loss_odom

		opt.zero_grad()
		loss.backward()
		opt.step()

		log_sample(i)

		if (i%10) == 0:
			val_loss_grid, val_loss_odom = validate(model)
			writer.add_scalar('CV Loss/occupancy', val_loss_grid, i)
			writer.add_scalar('CV Loss/odom', val_loss_odom, i)
finally:
	torch.save(model.state_dict(), 'model.pt')
