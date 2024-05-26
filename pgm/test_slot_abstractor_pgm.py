import os
import argparse
from slot_abstractor_pgm import *
import time
import datetime
import torch.optim as optim
import torch
from PIL import Image
from torchvision.transforms import transforms
from util import log
from torch.utils.data import Dataset, DataLoader
import json
from scipy import misc
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from socket import gethostname
import gc
from collections import OrderedDict
import torchvision.transforms.functional as TF
import random

def setup(rank, world_size):
	# initialize the process group
	dist.init_process_group("nccl", rank=rank, world_size=world_size)

class dataset(Dataset):
	def __init__(self, root_dir, dataset_type, img_size):
		self.root_dir = root_dir
		self.transforms = transforms.Compose(
			[
				transforms.ToTensor(),
				transforms.Lambda(lambda X: 2 * X - 1.0),  # rescale between -1 and 1
				transforms.Resize((img_size,img_size)),
			]
		)
				
		self.file_names = [root_dir + f for f in os.listdir(root_dir)
							if dataset_type in f]
			
		if dataset_type == 'train':

			
			self.train = True
		
		else:
		
			
			self.train=False


						
		self.img_size = img_size
		

	def __len__(self):
		return len(self.file_names)

	def __getitem__(self, idx):
		# data_path = os.path.join(self.root_dir, self.file_names[idx])
		data_path = self.file_names[idx]
		data = np.load(data_path)
		image = data["image"].reshape(16, 160, 160)
		target = data["target"]
		
		
		img = [self.transforms(Image.fromarray(image[i].astype(np.uint8))) for i in range(16)]
	
		
		return torch.stack(img,dim=0),target




def load_checkpoint(slot_model,abstractor_scoring_model,checkpoint_path):
	"""
	Loads weights from checkpoint
	:param model: a pytorch nn model
	:param str checkpoint_path: address/path of a file
	:return: pytorch nn model with weights loaded from checkpoint
	"""
	model_ckp = torch.load(checkpoint_path,map_location= torch.device('cpu'))
	
# create new OrderedDict that does not contain `module.`

	new_state_dict = OrderedDict()
	for k, v in model_ckp['slot_model_state_dict'].items():
		name = k[7:] # remove `module.`
		new_state_dict[name] = v
	# load params
	slot_model.load_state_dict(new_state_dict)
	# print("modules>>",len(model_ckp['slot_model_state_dict']))
	# slot_model.load_state_dict(model_ckp['slot_model_state_dict'])
	new_state_dict = OrderedDict()
	for k, v in model_ckp['abstractor_model_state_dict'].items():
		name = k[7:] # remove `module.`
		new_state_dict[name] = v
	# load params
	abstractor_scoring_model.load_state_dict(new_state_dict)
	# transformer_scoring_model.load_state_dict(model_ckp['transformer_scoring_model_state_dict'])
	
	return slot_model ,abstractor_scoring_model	


def load_abstractor_checkpoint(abstractor_scoring_model,checkpoint_path):
	"""
	Loads weights from checkpoint
	:param model: a pytorch nn model
	:param str checkpoint_path: address/path of a file
	:return: pytorch nn model with weights loaded from checkpoint
	"""
	model_ckp = torch.load(checkpoint_path,torch.device('cpu'))
	
# create new OrderedDict that does not contain `module.`

	
	# print("modules>>",len(model_ckp['slot_model_state_dict']))
	# slot_model.load_state_dict(model_ckp['slot_model_state_dict'])
	new_state_dict = OrderedDict()
	for k, v in model_ckp['abstractor_model_state_dict'].items():
		name = k[7:] # remove `module.`
		new_state_dict[name] = v
	# load params
	abstractor_scoring_model.load_state_dict(new_state_dict)
	# transformer_scoring_model.load_state_dict(model_ckp['transformer_scoring_model_state_dict'])
	
	return abstractor_scoring_model	

def load_slot_checkpoint(slot_model,checkpoint_path):
	"""
	Loads weights from checkpoint
	:param model: a pytorch nn model
	:param str checkpoint_path: address/path of a file
	:return: pytorch nn model with weights loaded from checkpoint
	"""
	model_ckp = torch.load(checkpoint_path,map_location= torch.device('cpu'))


	slot_model.load_state_dict(model_ckp['slot_model_state_dict'])
	# transformer_scoring_model.load_state_dict(model_ckp['transformer_scoring_model_state_dict'])
	
	return slot_model 	

	



parser = argparse.ArgumentParser()

parser.add_argument
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_slots', default=16, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
parser.add_argument('--hid_dim', default=32, type=int, help='hidden dimension size')
parser.add_argument('--depth', default=6, type=int, help='transformer number of layers')
parser.add_argument('--heads', default=8, type=int, help='transformer number of heads')
parser.add_argument('--mlp_dim', default=512, type=int, help='transformer mlp dimension')

parser.add_argument('--learning_rate', default=0.0004, type=float)
parser.add_argument('--warmup_steps', default=10000, type=int, help='Number of warmup steps for the learning rate.')
parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
parser.add_argument('--decay_steps', default=100000, type=int, help='Number of steps for the learning rate decay.')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
parser.add_argument('--num_epochs', default=1, type=int, help='number of workers for loading data')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--run', type=str, default='1')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--save_interval', type=int, default=1000)

parser.add_argument('--path', default='pgm_datasets/extrapolation/', type=str, help='dataset path')
parser.add_argument('--img_size', type=int, default=128)


parser.add_argument('--model_checkpoint', type=str, help = 'path where model weights are saved', required= True)
parser.add_argument('--apply_context_norm', type=bool, default=True)

# parser.add_argument('--accumulation_steps', type=int, default=8)

opt = parser.parse_args()
print(opt)
use_cuda = not opt.no_cuda and torch.cuda.is_available()
# device = torch.device("cuda:" + str(opt.device) if use_cuda else "cpu")

world_size    = int(os.environ["WORLD_SIZE"])
rank          = int(os.environ["SLURM_PROCID"])
gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
assert gpus_per_node == torch.cuda.device_count()
print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
	  f" {gpus_per_node} allocated GPUs per node.", flush=True)

dist.init_process_group("nccl", rank=rank, world_size=world_size)
if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)




local_rank = rank - gpus_per_node * (rank // gpus_per_node)
torch.cuda.set_device(local_rank)
print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")


log.info('Loading pgm datasets...')
train_data = dataset(opt.path, "train", opt.img_size)
valid_data = dataset(opt.path, "val", opt.img_size )
test_data = dataset(opt.path, "test", opt.img_size )

train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=world_size, rank=rank)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size, sampler=train_sampler, 
											   num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),  pin_memory=True)


val_sampler = torch.utils.data.distributed.DistributedSampler(valid_data, num_replicas=world_size, rank=rank)
val_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=opt.batch_size, sampler=val_sampler, 
											   num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),  pin_memory=True)


test_sampler = torch.utils.data.distributed.DistributedSampler(test_data, num_replicas=world_size, rank=rank)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size, sampler=test_sampler, 
											   num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),  pin_memory=True)




print("Number of samples in val set>>",len(val_dataloader))
print("Number of samples in test set>>",len(test_dataloader))

log.info('Building model...')
slot_model = SlotAttentionAutoEncoder((opt.img_size,opt.img_size), opt.num_slots, opt.num_iterations, opt.hid_dim)


abstractor_scoring_model = scoring_model(opt,opt.hid_dim,opt.depth,opt.heads,opt.mlp_dim,opt.num_slots)
slot_model = load_slot_checkpoint(slot_model,'slot_attention_autoencoder_16slots_dspritesdecoder_morewarmup_lrdecay_pgm_wholetrainset_neutral_run_1_best.pth.tar')

abstractor_scoring_model = load_abstractor_checkpoint(abstractor_scoring_model,opt.model_checkpoint)
slot_model = slot_model.to(local_rank)


ddp_abstractor_model = DDP(abstractor_scoring_model.to(local_rank), device_ids=[local_rank])


mse_criterion = nn.MSELoss()
ce_criterion = nn.CrossEntropyLoss()


log.info('Testing begins...')
start = time.time()
i = 0
# optimizer.zero_grad()
slot_model.eval()
	
ddp_abstractor_model.eval()

all_testacc=[]

# ddp_slot_model.eval()


for batch_idx, (img,target) in enumerate(test_dataloader):
	# print("image and target shape>>",img.shape,target.shape)
# 		# print(torch.max(img), torch.min(img),img.shape)
	
	img = img.to(local_rank) #.unsqueeze(1).float()
	target = target.to(local_rank)
	feat_slots_seq =[]
	pos_slots_seq =[]
	recon_combined_seq =[]
	recons_seq=[]
	masks_seq=[]
	for idx in range(img.shape[1]):
		# print(idx)

	
		# recon_combined, recons, masks, slots,_ = ddp_slot_model(img[:,idx],local_rank)
		recon_combined, recons, masks, feat_slots,pos_slots,_= slot_model(img[:,idx],local_rank)
		
		# slots = ddp_slot_model(img[:,idx],local_rank)
		
		feat_slots_seq.append(feat_slots)
		pos_slots_seq.append(pos_slots)
		recon_combined_seq.append(recon_combined)
		recons_seq.append(recons)
		masks_seq.append(masks)
		del recon_combined,recons, masks, feat_slots,pos_slots


	given_panels_feats = torch.stack(feat_slots_seq,dim=1)[:,:8]
	given_panels_pos = torch.stack(pos_slots_seq,dim=1)[:,:8]
	answer_panels_feats = torch.stack(feat_slots_seq,dim=1)[:,8:]
	answer_panels_pos = torch.stack(pos_slots_seq,dim=1)[:,8:]
	# print("given and answer panels shape>>>", given_panels.shape, answer_panels.shape)

	scores = ddp_abstractor_model(given_panels_feats,given_panels_pos,answer_panels_feats,answer_panels_pos,local_rank)
	# print("scores and target>>",scores,target)
	pred = scores.argmax(1)
	acc = torch.eq(pred,target).float().mean().item() * 100.0
	
	
	all_testacc.append(acc)
	if batch_idx % opt.log_interval == 0:
		log.info(
				 '[Batch: ' + str(batch_idx) + ' of ' + str(len(test_dataloader)) + '] ' + \
				 
				  '[MSE Loss = ' + '{:.4f}'.format(mse_criterion(torch.stack(recon_combined_seq,dim=1), img).item()) + '] ' +\
				 '[CE Loss = ' + '{:.4f}'.format(ce_criterion(scores,target).item()) + '] ' +\
				 '[Accuracy = ' + '{:.2f}'.format(acc) + ']'

				  )

print("Average test accuracy>>",np.mean(np.array(all_testacc)))

	

