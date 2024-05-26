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
		
		img = [self.transforms(Image.fromarray(cv2.cvtColor(cv2.imread(os.readlink("/".join(data_path.split("/")[-5:])+'/{}.jpg'.format(i))[15:]),cv2.COLOR_BGR2RGB).astype(np.uint8))) for i in range(1,17)]
	
		target = self.extract_labels(data_path.split("/")[-1])

		return torch.stack(img,dim=0), target


def save_slot_abstractor_nw(ddp_slot_model,ddp_abstractor_model,optimizer,epoch, name,save_path):

	
	torch.save({
		'slot_model_state_dict': ddp_slot_model.state_dict(),
		'abstractor_model_state_dict': ddp_abstractor_model.state_dict(),
		
		
		'optimizer_state_dict': optimizer.state_dict(),
		'epoch': epoch,
	}, save_path+name)


def load_slot_checkpoint(slot_model,checkpoint_path):
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
	for k, v in model_ckp['slot_model_state_dict'].items():
		name = k[7:] # remove `module.`
		new_state_dict[name] = v
	# load params
	slot_model.load_state_dict(new_state_dict)
	# transformer_scoring_model.load_state_dict(model_ckp['transformer_scoring_model_state_dict'])
	
	return slot_model	



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
parser.add_argument('--num_epochs', default=100, type=int, help='number of workers for loading data')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--run', type=str, default='1')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--save_interval', type=int, default=10)
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--save_path', default='weights/', type=str, help='model save path')

# parser.add_argument('--model_name', type=str, default='slot_attention_random_spatial_heldout_unicodes_resizedcropped_pretrained_frozen_autoencoder_new_correlnet-T_scoring')
parser.add_argument('--model_name', type=str, default='dinosaur_vitb16_pretrained_slotattn_mlpdecoder_abstractor_multigpu')

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

train_data = dataset("neutral_train_problems.npy", args.img_size)
val_data = dataset("neutral_val_problems.npy", args.img_size)

train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=world_size, rank=rank)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler, 
											   num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),  pin_memory=True)

val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, num_replicas=world_size, rank=rank)
val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, sampler=val_sampler, 
											   num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),  pin_memory=True)






print("Number of samples in training set>>",len(train_dataloader))
print("Number of samples in validation set>>",len(val_dataloader))

log.info('Building model...')

model = timm.create_model('vit_base_patch16_224.dino',pretrained=True, pretrained_cfg_overlay = dict(file='vit_base_patch16_224.dino/model.safetensors'))
model = model.to(local_rank)

vit_orig_pos_embed = model.pos_embed[:,1:]

slot_model = SlotAttentionAutoEncoder(196, 768, args.num_slots,args.num_iterations, args.hid_dim)
slot_model = load_slot_checkpoint(slot_model, 'dinosaur_vitb16_encoder_mlpdecoder_multigpu_11slots_warmup_vprom_neutral_trainprobs_reconimgs_run_1_best.pth.tar')

for param in model.parameters():
	param.requires_grad = False

# for param in slot_model.parameters():
# 	param.requires_grad = False
ddp_slot_model = DDP(slot_model.to(local_rank), device_ids=[local_rank])

abstractor_scoring_model = scoring_model(args,196,args.hid_dim,args.depth,args.heads,args.mlp_dim,args.num_slots)
ddp_abstractor_model = DDP(abstractor_scoring_model.to(local_rank), device_ids=[local_rank])

mse_criterion = nn.MSELoss()
ce_criterion = nn.CrossEntropyLoss()


params = [{'params': list(ddp_slot_model.parameters()) + list(ddp_abstractor_model.parameters())}]
# params = [{'params': ddp_abstractor_model.parameters()}]

log.info('Setting up optimizer...')

optimizer = optim.Adam(params, lr=args.learning_rate)

log.info('Training begins...')
i = 0 
max_val_acc=0
for epoch in range(1,args.num_epochs+1):
	train_sampler.set_epoch(epoch)
	model.eval()
	ddp_slot_model.train()
	ddp_abstractor_model.train()

	
	# all_trainloss = []
	all_trainmseloss = []

	all_trainceloss=[]
	all_trainacc = []
	all_valacc=[]

	for batch_idx, (img,target) in enumerate(train_dataloader):
		
		learning_rate = args.learning_rate
		# learning_rate = learning_rate * (opt.decay_rate ** (
		# 	i / opt.decay_steps))

		optimizer.param_groups[0]['lr'] = learning_rate

		img = img.to(local_rank).float()
		target = target.to(local_rank)
		
		recon_combined_seq =[]
		feats_seq = []
		slotfeats_seq =[]
		attn_seq =[]
		
		masks_seq = []
		for idx in range(img.shape[1]):

			feats = model.forward_features(img[:,idx])[:,1:]
			
			recon_combined, recons, masks, feat_slots,attn = ddp_slot_model(feats,local_rank)
			recon_combined_seq.append(recon_combined)
			masks_seq.append(masks)
			feats_seq.append(feats)
			slotfeats_seq.append(feat_slots)
			attn_seq.append(attn)

			del recon_combined,recons, masks, feat_slots,attn
		# print("reconstructed features and dino feats shape>>",torch.stack(recon_combined_seq,dim=1).shape,torch.stack(feats_seq,dim=1).shape)
		given_panels_feats = torch.stack(slotfeats_seq,dim=1)[:,:8]
	
		answer_panels_feats = torch.stack(slotfeats_seq,dim=1)[:,8:]
		

		scores = ddp_abstractor_model(given_panels_feats,answer_panels_feats,vit_orig_pos_embed,torch.stack(attn_seq,dim=1),local_rank)
		# print("scores and target>>",scores,target)
		pred = scores.argmax(1)
		acc = torch.eq(pred,target).float().mean().item() * 100.0

		loss = 10*mse_criterion(torch.stack(recon_combined_seq,dim=1),torch.stack(feats_seq,dim=1)) + ce_criterion(scores,target)

		all_trainmseloss.append(mse_criterion(torch.stack(recon_combined_seq,dim=1),torch.stack(feats_seq,dim=1)).item())
		all_trainceloss.append(ce_criterion(scores,target).item())
		
		all_trainacc.append(acc)
# 		del recons, masks, slots
		# loss = loss / opt.accumulation_steps   
		optimizer.zero_grad()
		loss.backward()
	

		optimizer.step()

		

		# print("learning rate>>>",learning_rate)
		# for j, para in enumerate(slotmask_model.parameters()):
		#     print(f'{j + 1}th parameter tensor:', para.shape)
		#     # print(para)
		#     print("gradient>>",para.grad)
		# if (batch_idx+1) % opt.accumulation_steps == 0:             # Wait for several backward steps
		# 	optimizer.step()                            # Now we can do an optimizer step
		# 	optimizer.zero_grad()  
		
		if batch_idx % args.log_interval == 0:
			log.info('[Epoch: ' + str(epoch) + '] ' + \
					 '[Batch: ' + str(batch_idx) + ' of ' + str(len(train_dataloader)) + '] ' + \
					 '[Total Loss = ' + '{:.4f}'.format(loss.item()) + '] ' +\
					 '[MSE Loss = ' + '{:.4f}'.format(mse_criterion(torch.stack(recon_combined_seq,dim=1),torch.stack(feats_seq,dim=1)).item()) + '] ' +\
					 '[CE Loss = ' + '{:.4f}'.format(ce_criterion(scores,target).item()) + '] ' +\
					 '[Learning rate = ' + '{:.8f}'.format(learning_rate) + '] ' + \
					 '[Accuracy = ' + '{:.2f}'.format(acc) + ']'
					  )

	print("Average training reconstruction loss>>",np.mean(np.array(all_trainmseloss)))
	print("Average training cross entropy loss>>",np.mean(np.array(all_trainceloss)))
	print("Average training accuracy>>",np.mean(np.array(all_trainacc)))

	ddp_slot_model.eval()
	ddp_abstractor_model.eval()
	
	for val_batch_idx, (val_img,val_target) in enumerate(val_dataloader):
		

		# learning_rate = learning_rate * (opt.decay_rate ** (
		# 	i / opt.decay_steps))

		

		val_img = val_img.to(local_rank).float()
		val_target = val_target.to(local_rank)
		val_recon_combined_seq =[]
		val_feats_seq = []
		val_slotfeats_seq =[]
		val_attn_seq =[]
		for idx in range(val_img.shape[1]):

			val_feats = model.forward_features(val_img[:,idx])[:,1:]
			
			val_recon_combined, val_recons, val_masks, val_feat_slots, val_attn = ddp_slot_model(val_feats,local_rank)
			val_recon_combined_seq.append(val_recon_combined)
			val_feats_seq.append(val_feats)
			val_slotfeats_seq.append(val_feat_slots)
			val_attn_seq.append(val_attn)

			del val_recon_combined,val_recons, val_masks, val_feat_slots, val_attn
		# print("reconstructed features>>",recon_combined.shape)

		given_panels_feats = torch.stack(val_slotfeats_seq,dim=1)[:,:8]
		
		answer_panels_feats = torch.stack(val_slotfeats_seq,dim=1)[:,8:]
		

		scores = ddp_abstractor_model(given_panels_feats,answer_panels_feats,vit_orig_pos_embed,torch.stack(val_attn_seq,dim=1),local_rank)
		# print("scores and target>>",scores,target)
		pred = scores.argmax(1)
		acc = torch.eq(pred,val_target).float().mean().item() * 100.0
# 		# print(torch.max(recon_combined), torch.min(recon_combined))
# 		# print(img.shape, recon_combined.shape, recons.shape, masks.shape, slots.shape)
		# if batch_idx<10:
		# 	total_train_images = torch.cat((total_train_images,img),dim=0)
		# 	total_train_recons_combined = torch.cat((total_train_recons_combined,torch.stack(recon_combined_seq,dim=1)),dim=0)
		# 	total_train_recons = torch.cat((total_train_recons,torch.stack(recons_seq,dim=1)),dim=0)
		# 	total_train_masks = torch.cat((total_train_masks,torch.stack(masks_seq,dim=1)),dim=0)
			
		
		all_valacc.append(acc)

		
	print("Average validation accuracy>>",np.mean(np.array(all_valacc)))
	if np.mean(np.array(all_valacc)) > max_val_acc:
		print("Validation accuracy increased from %s to %s"%(max_val_acc,np.mean(np.array(all_valacc))))
		max_val_acc = np.mean(np.array(all_valacc))
		print("Saving model$$$$")
		save_slot_abstractor_nw(ddp_slot_model,ddp_abstractor_model,optimizer,epoch,'{}_10weightmse_assymm_qkweights_initsame_layerwise_original_vitposcodes_0.1dropout_tcn_11slots_lowerlr_nolrdecay_rowcolposemb_added_vprom_neutral_run_{}_best.pth.tar'.format(args.model_name,args.run),args.save_path)
		# save_transformer_nw(ddp_transformer_scoring_model,optimizer,epoch,'{}_tcn_16slots_dspritesdecoder_nolrdecay_rowcolposemb_pgm_wholetrainset_extrapolation_run_{}_best.pth.tar'.format(opt.model_name,opt.run))
		
		# save_transformer_nw(ddp_transformer_scoring_model,optimizer,epoch,'{}_tcn_16slots_dspritesdecoder_nolrdecay_rowcolposemb_pgm_wholetrainset_neutral_run_{}_best.pth.tar'.format(opt.model_name,opt.run))
	
	else:
		print("Validation accuracy didn't increase and hence skipping model saving!!!!")
	print("Best validation accuracy till now >>",max_val_acc)
	
dist.destroy_process_group()
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
		











