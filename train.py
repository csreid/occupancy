from tqdm import tqdm
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
import torch
from torch.optim import Adam
from dataloader import OccupancyDataLoader
from model import OccupancyGridModule
from torch.utils.tensorboard import SummaryWriter

BATCH_SIZE=8
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

opt = Adam(model.parameters(), weight_decay=0.1)
loader = OccupancyDataLoader(cv_fraction=0.1)

loss_fn_grid = BCEWithLogitsLoss()
loss_fn_odom = MSELoss()

def validate(model):
	with torch.no_grad():
		imgs, scans, grids = loader.sample(4, for_cv=True)
		pred = model(scans.to(dev), imgs.to(dev)).squeeze().flatten(start_dim=0, end_dim=1)
		loss = loss_fn(
				pred,
				grids.flatten(start_dim=0, end_dim=1).to(dev),
		)
		return loss

def log_sample(i):
	sample_imgs, sample_scans, sample_grid = loader.sample(1, for_cv=True)
	with torch.no_grad():
		sample_out = model(sample_scans.to(dev), sample_imgs.to(dev)).squeeze().unsqueeze(0).unsqueeze(2)
		sample_grid = sample_grid.squeeze().unsqueeze(0).unsqueeze(2).to(dev)

		video = torch.cat((sample_grid, sample_out), dim=-1).expand(-1, -1, 3, -1, -1) # Pop the one channel into RGB to treat it like a video

		writer.add_video('Sample', video, i)

try:
	for i in tqdm(range(ITERATIONS)):
		#imgs, scans, grids = loader.sample(16)
		(imgs, scans, init_pose), (grids, poses) = loader.sample(2)
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
		writer.add_scalar('Occupancy Grid Loss', loss_grid, i)
		writer.add_scalar('Odom Loss', loss_odom, i)

		loss = loss_grid + loss_odom

		opt.zero_grad()
		loss.backward()
		opt.step()

		if (i%1) == 0:
			log_sample(i)
			val_loss = validate(model)
			writer.add_scalar('CV Loss', val_loss, i)
finally:
	torch.save(model.state_dict(), 'model.pt')
