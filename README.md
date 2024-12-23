
# Slot Abstractors
Official repository for the paper - "[Slot Abstractors: Toward Scalable Abstract Visual Reasoning](https://arxiv.org/pdf/2403.03458)" 

## Requirements 
- python 3.9.7
- NVIDIA GPU with CUDA 11.0+ capability
- torch==1.13.1
- torchvision==0.14.1
- glob
- PIL==8.4.0
- numpy==1.23.1
- einops==0.4.1
- timm==0.9.5
- opencv-python==4.7.0
- json==2.0.9

### Note - For training/testing, jobs were run through the Slurm scheduler. 

## ART
First `cd art`

Then execute `python run_art_tasks.py` which trains the slot abstractor on the four ART tasks (same/different (sd), relational-match-to-sample (rmts), distribution-of-3 (dist3), identity rules (idrules)) by running independent jobs for each task, and saves the test accuracy for each task and model run in a txt file.

For example, the saved path for the test accuracy file for same/different task corresponding to model run number 1 would be `test/same_diff/m95/slot_attention_random_spatial_heldout_unicodes_resizedcropped_pretrained_frozen_autoencoder_abstractor_scoring/run1.txt`

## SVRT

First `cd svrt`

Generate the SVRT dataset using the instructions in https://github.com/Shanka123/OCRA 

Then execute `python run_svrt_tasks.py` which trains the slot abstractor on all the 23 SVRT tasks by running independent jobs for each task with 500 and 1000 training samples, and saves the test accuracy for each task in a txt file.

For example, the saved path for the test accuracy file for task number 1, trained with 500 samples would be `test/svrt/slot_attention_augmentations_first_more_pretrained_svrt_alltasks_frozen_autoencoder_abstractor_scoring/500/results_problem_1.txt`

## CLEVR-ART

First `cd clevr-art`

Generate the CLEVR-ART dataset using the instructions in https://github.com/Shanka123/OCRA

Then execute `python run_clevr_tasks.py` which trains the slot abstractor on the CLEVR-ART tasks (relational-match-to-sample (rmts) and identity rules (idrules)) by running independent jobs for each task, and saves the test accuracy for each task and model run in a txt file.

For example, the saved path for the test accuracy file for relational-match-to-sample task corresponding to model run number 1 would be `test/CLEVR_RMTS/slot_attention_random_spatial_clevrshapes_cv2_rgbcolororder_pretrained_frozen_autoencoder_abstractor_scoring/run1.txt`


## PGM

First `cd pgm`

Create a directory `pgm_datasets` and download the dataset in it from https://github.com/google-deepmind/abstract-reasoning-matrices. Also, create a separate directory `weights` to save model weights from training.

Then execute `sbatch run_train_pgm_job.slurm` which trains the slot abstractor on the extrapolation regime by default. To train on a different regime modify `--path` argument on line 23, and the name of the output log file on line 10.

To evaluate on the test set (default is extrapolation regime), execute `sbatch run_test_pgm_job.slurm`, which requires specifying the path to the saved model weights after training through `--model_checkpoint` argument on line 22.


## V-PROM

First `cd v-prom`

Create a directory `V-PROM` and download the dataset in it. Also, create a separate directory `weights` to save model weights from training.

Download the `vit_base_patch16_224.dino` model using the timm library, and the slot attention model pre-trained on the V-PROM neutral regime from [here](https://drive.google.com/file/d/1JMJ-29TKygrNZYnqKna2DmaiOaY5iKvd/view?usp=drive_link). 

Then execute `sbatch run_train_dinosaur_abstractor_vprom_job.slurm` which trains the slot abstractor on the V-PROM neutral regime. 

To evaluate on the test set, execute `sbatch run_test_dinosaur_abstractor_vprom_job.slurm`, which requires specifying the path to the saved model weights after training through `--model_checkpoint` argument on line 24.

## Citation

We thank you for showing interest in our work. If our work was beneficial for you, please consider citing us using:
```
@inproceedings{mondal2024slot,
title={Slot Abstractors: Toward Scalable Abstract Visual Reasoning},
author={Mondal, Shanka Subhra and Cohen, Jonathan D and Webb, Taylor W},
booktitle={Fourty-first International Conference on Machine Learning},
year={2024}
}


