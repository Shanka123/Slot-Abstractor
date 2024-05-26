import timm
import numpy as np
from PIL import Image
from dinosaur_abstractor_vprom import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import os
import argparse
import json
from util import log
import cv2
import math
from collections import OrderedDict
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from socket import gethostname

def setup(rank, world_size):
	# initialize the process group
	dist.init_process_group("nccl", rank=rank, world_size=world_size)

class dataset(Dataset):
	def __init__(self, data, img_size):
	
		self.transforms = transforms.Compose(
			[	
				transforms.Resize((img_size,img_size)),
				transforms.ToTensor(),
				
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
				
			]
		)
				
		self.file_names = np.load(data)
		self.obj_gt_labels = np.load("V-PROM/datasets/obj_gt_labels.npy",allow_pickle=True).item()
		self.attr_gt_labels  = np.load("V-PROM/datasets/attr_gt_labels.npy",allow_pickle=True).item()
		self.counting_gt_labels  = np.load("V-PROM/datasets/counting_gt_labels.npy",allow_pickle=True).item()
			 
		# print(self.file_names[:100])
		# if dataset_type == 'train':
		# 	self.file_names = self.file_names[:10000]	
							
		self.img_size = img_size
		# self.embeddings = np.load('./embedding.npy')

	def __len__(self):
		return len(self.file_names)

	def extract_labels(self,problem_number):
		

		if problem_number in self.obj_gt_labels['and'].keys():
		    
		    
		    label = self.obj_gt_labels['and'][problem_number]['label']

		elif problem_number in self.obj_gt_labels['or'].keys():
		    
		    label = self.obj_gt_labels['or'][problem_number]['label']

		    
		elif problem_number in self.obj_gt_labels['union'].keys():
		    
		    
		    label = self.obj_gt_labels['union'][problem_number]['label']

		elif problem_number in self.attr_gt_labels['and']['obj'].keys():
		    
		    label = self.attr_gt_labels['and']['obj'][problem_number]['label']

		elif problem_number in self.attr_gt_labels['or']['obj'].keys():
		    
		    label = self.attr_gt_labels['or']['obj'][problem_number]['label']
		    
		elif problem_number in self.attr_gt_labels['union']['obj'].keys():
		    
		    label = self.attr_gt_labels['union']['obj'][problem_number]['label']
		    
		elif problem_number in self.attr_gt_labels['and']['people'].keys():
		    
		    label = self.attr_gt_labels['and']['people'][problem_number]['label']

		elif problem_number in self.attr_gt_labels['or']['people'].keys():
		    
		    label = self.attr_gt_labels['or']['people'][problem_number]['label']
		    
		elif problem_number in self.attr_gt_labels['union']['people'].keys():
		   
		    label = self.attr_gt_labels['union']['people'][problem_number]['label']

		elif problem_number in self.counting_gt_labels['and'].keys():
		    
		    label = self.counting_gt_labels['and'][problem_number]['label']

		elif problem_number in self.counting_gt_labels['or'].keys():
		    
		    label = self.counting_gt_labels['or'][problem_number]['label']

		elif problem_number in self.counting_gt_labels['union'].keys():
		    
		    label = self.counting_gt_labels['union'][problem_number]['label']

		elif problem_number in self.counting_gt_labels['progression'].keys():
		    
		    label = self.counting_gt_labels['progression'][problem_number]['label']

		
		return label-9


		    


	def __getitem__(self, idx):
		# data_path = os.path.join(self.root_dir, self.file_names[idx])
		data_path = self.file_names[idx]
		

	
		# resize_image = misc.imresize(data["image"][:,:,0], (self.img_size, self.img_size))
		
		# img = [self.transforms(Image.fromarray(cv2.cvtColor(cv2.imread('/scratch/gpfs/smondal/' + os.readlink(data_path+'/{}.jpg'.format(i))[15:]),cv2.COLOR_BGR2RGB).astype(np.uint8))) for i in range(1,17)]
		
		img = [self.transforms(Image.fromarray(cv2.cvtColor(cv2.imread(os.readlink("/".join(data_path.split("/")[-5:])+'/{}.jpg'.format(i))[15:]),cv2.COLOR_BGR2RGB).astype(np.uint8))) for i in range(1,17)]
	
		target = self.extract_labels(data_path.split("/")[-1])

		return torch.stack(img,dim=0), target


	

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


parser = argparse.ArgumentParser()

parser.add_argument
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_slots', default=11, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
parser.add_argument('--hid_dim', default=256, type=int, help='hidden dimension size')
parser.add_argument('--depth', default=6, type=int, help='transformer number of layers')
parser.add_argument('--heads', default=8, type=int, help='transformer number of heads')
parser.add_argument('--mlp_dim', default=512, type=int, help='transformer mlp dimension')

parser.add_argument('--learning_rate', default=4e-4, type=float)
parser.add_argument('--warmup_steps', default=10000, type=int, help='Number of warmup steps for the learning rate.')
parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
parser.add_argument('--decay_steps', default=100000, type=int, help='Number of steps for the learning rate decay.')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
parser.add_argument('--num_epochs', default=1, type=int, help='number of workers for loading data')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--run', type=str, default='1')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--save_interval', type=int, default=10)
parser.add_argument('--img_size', type=int, default=224)



parser.add_argument('--model_checkpoint', type=str, default='model saved checkpoint')


parser.add_argument('--apply_context_norm', type=bool, default=True)

args = parser.parse_args()
print(args)
use_cuda = not args.no_cuda and torch.cuda.is_available()
# device = torch.device("cuda:" + str(args.device) if use_cuda else "cpu")


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


log.info('Preparing data...')

test_data = dataset("neutral_test_problems.npy", args.img_size)


test_sampler = torch.utils.data.distributed.DistributedSampler(test_data, num_replicas=world_size, rank=rank)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, sampler=test_sampler, 
											   num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),  pin_memory=True)


								  






print("Number of samples in test set>>",len(test_dataloader))


log.info('Building model...')

model = timm.create_model('vit_base_patch16_224.dino',pretrained=True, pretrained_cfg_overlay = dict(file='vit_base_patch16_224.dino/model.safetensors'))
model = model.to(local_rank)

vit_orig_pos_embed = model.pos_embed[:,1:]

slot_model = SlotAttentionAutoEncoder(196, 768, args.num_slots,args.num_iterations, args.hid_dim)


# for param in slot_model.parameters():
# 	param.requires_grad = False

abstractor_scoring_model = scoring_model(args,196,args.hid_dim,args.depth,args.heads,args.mlp_dim,args.num_slots)
slot_model, abstractor_model = load_checkpoint(slot_model, abstractor_scoring_model,args.model_checkpoint)

ddp_slot_model = DDP(slot_model.to(local_rank), device_ids=[local_rank])
ddp_abstractor_model = DDP(abstractor_model.to(local_rank), device_ids=[local_rank])



mse_criterion = nn.MSELoss()
ce_criterion = nn.CrossEntropyLoss()



log.info('Testing begins...')
i = 0 


all_testacc = []	
		

model.eval()	
ddp_slot_model.eval()
ddp_abstractor_model.eval()

for batch_idx, (img,target) in enumerate(test_dataloader):
	

	# learning_rate = learning_rate * (opt.decay_rate ** (
	# 	i / opt.decay_steps))

	

	img = img.to(local_rank).float()
	target = target.to(local_rank)
	recon_combined_seq =[]
	feats_seq = []
	slotfeats_seq =[]
	attn_seq =[]
	for idx in range(img.shape[1]):

		feats = model.forward_features(img[:,idx])[:,1:]
		
		recon_combined, recons, masks, feat_slots, attn = ddp_slot_model(feats,local_rank)
		recon_combined_seq.append(recon_combined)
		feats_seq.append(feats)
		slotfeats_seq.append(feat_slots)
		attn_seq.append(attn)

		del recon_combined,recons, masks, feat_slots, attn
	# print("reconstructed features>>",recon_combined.shape)

	given_panels_feats = torch.stack(slotfeats_seq,dim=1)[:,:8]
	
	answer_panels_feats = torch.stack(slotfeats_seq,dim=1)[:,8:]
	

	scores = ddp_abstractor_model(given_panels_feats,answer_panels_feats,vit_orig_pos_embed,torch.stack(attn_seq,dim=1),local_rank)
	# print("scores and target>>",scores,target)
	pred = scores.argmax(1)
	acc = torch.eq(pred,target).float().mean().item() * 100.0
# 		# print(torch.max(recon_combined), torch.min(recon_combined))
# 		# print(img.shape, recon_combined.shape, recons.shape, masks.shape, slots.shape)
	# if batch_idx<10:
	# 	total_train_images = torch.cat((total_train_images,img),dim=0)
	# 	total_train_recons_combined = torch.cat((total_train_recons_combined,torch.stack(recon_combined_seq,dim=1)),dim=0)
	# 	total_train_recons = torch.cat((total_train_recons,torch.stack(recons_seq,dim=1)),dim=0)
	# 	total_train_masks = torch.cat((total_train_masks,torch.stack(masks_seq,dim=1)),dim=0)
		
	
	all_testacc.append(acc)

	if batch_idx % args.log_interval == 0:
		log.info(
				 '[Batch: ' + str(batch_idx) + ' of ' + str(len(test_dataloader)) + '] ' + \
				 
				  '[MSE Loss = ' + '{:.4f}'.format(mse_criterion(torch.stack(recon_combined_seq,dim=1),torch.stack(feats_seq,dim=1)).item()) + '] ' +\
				 '[CE Loss = ' + '{:.4f}'.format(ce_criterion(scores,target).item()) + '] ' +\
				 '[Accuracy = ' + '{:.2f}'.format(acc) + ']'

				  )

	
print("Average test accuracy>>",np.mean(np.array(all_testacc)))


		# print("learning rate>>>",learning_rate)
		# for j, para in enumerate(slotmask_model.parameters()):
		#     print(f'{j + 1}th parameter tensor:', para.shape)
		#     # print(para)
		#     print("gradient>>",para.grad)
		# if (batch_idx+1) % opt.accumulation_steps == 0:             # Wait for several backward steps
		# 	optimizer.step()                            # Now we can do an optimizer step
		# 	optimizer.zero_grad()  
		
	


	# if epoch % args.save_interval==0:

	# 	save_nw(slot_model,optimizer,epoch,'{}_7slots_bongard_hoi_all_train_images_run_{}.pth.tar'.format(args.model_name,args.run))
			

	# np.savez('predictions/train_original_images_masks.npz', images= orig_img.cpu().detach().numpy(),masks = masks.cpu().detach().numpy())
		











