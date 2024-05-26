import os

configurations = ['results_problem_1','results_problem_2','results_problem_3','results_problem_4','results_problem_5','results_problem_6','results_problem_7','results_problem_8','results_problem_9','results_problem_10','results_problem_11','results_problem_12','results_problem_13','results_problem_14','results_problem_15','results_problem_16','results_problem_17','results_problem_18','results_problem_19','results_problem_20','results_problem_21','results_problem_22','results_problem_23' ]
n_samples = [500,1000]


for run_no in range(1,2):

	for config in configurations:
		for num_data in n_samples:
			if num_data==500:

		


	
			
				os.system("sbatch -N 1 -n 8 -t 1000 -o slot_attention_augmentations_first_more_pretrained_svrt_alltasks_500_images_frozen_autoencoder_6slots_lowerlr_64dim_128res_depth=24_abstractor_scoring_svrt_%s_250perclass_run%s.log --gres=gpu:1 --wrap 'python -u train_slot_abstractor_svrt.py  --batch_size 32  --img_size 128 --learning_rate 4e-5 --n 250 --model_checkpoint 'slot_attention_autoencoder_augmentations_6slots_clevrdecoder_morewarmup_lowerlr_nolrdecay_64dim_128res_grayscale_svrt_alltasks_num_images_250_run_1_more_x3_continuetraining_best.pth.tar'  --configuration %s --run %s  '"%(config,run_no,config,str(run_no)))
		
			else:

				os.system("sbatch -N 1 -n 8 -t 1400 -o slot_attention_augmentations_first_more_pretrained_svrt_alltasks_1000_images_frozen_autoencoder_6slots_lowerlr_64dim_128res_abstractor_scoring_svrt_%s_500perclass_run%s.log --gres=gpu:1 --wrap 'python -u train_slot_abstractor_svrt.py  --batch_size 32 --learning_rate 4e-5  --img_size 128 --n 500 --model_checkpoint 'slot_attention_autoencoder_augmentations_6slots_clevrdecoder_morewarmup_lowerlr_nolrdecay_64dim_128res_grayscale_svrt_alltasks_num_images_500_run_1_more_more_continuetraining_best.pth.tar'  --configuration %s --run %s  '"%(config,run_no,config,str(run_no)))
		
		


	

