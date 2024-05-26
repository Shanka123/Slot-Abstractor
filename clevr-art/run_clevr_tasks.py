import os
tasks = ['rmts','id']


for run_no in range(1,6):
	for t in tasks:
		if t=='rmts':
			os.system("sbatch -N 1 -n 8 -t 200 -o slot_attention_pretrained_random_spatial_clevrshapes_cv2_rgbcolororder_frozen_autoencoder_7slots_nowarmup_lowerlr_nolrdecay_64dim_128res_cv2_rgbcolororder_depth=24_abstractor_scoring_rmts_ood_run%s.log --gres=gpu:1 --constraint=gpu80  --wrap 'python -u train_slot_abstractor_clevr_rmts.py  --img_size 128 --run %s   '"%(run_no,str(run_no)))
	
		else:

			os.system("sbatch -N 1 -n 8 -t 1500 -o slot_attention_pretrained_random_spatial_clevrshapes_cv2_rgbcolororder_frozen_autoencoder_7slots_nowarmup_lowerlr_nolrdecay_64dim_128res_cv2_rgbcolororder_depth=24_abstractor_scoring_idrules_ood_run%s.log --gres=gpu:1 --constraint=gpu80  --wrap 'python -u train_slot_abstractor_clevr_idrules.py  --img_size 128 --run %s  '"%(run_no,str(run_no)))
	
