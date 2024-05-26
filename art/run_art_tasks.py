import os


holdout = [95]

tasks = ['sd','rmts','dist3', 'id']
for run_no in range(1,11):

	for t in tasks:
		for m in holdout:


	
			if t=='sd' and m==95:

				os.system("sbatch -N 1 -n 8 -t 200 -o slot_attention_pretrained_heldout_unicodes_frozen_autoencoder_6slots_nowarmup_lowerlr_nolrdecay_64dim_128res_abstractor_scoring_sd_holdout%s_run%s.log --gres=gpu:1 --constraint=gpu80 --wrap 'python -u train_slot_abstractor_sd.py  --batch_size 16  --img_size 128 --num_epochs 600  --m_holdout %s --run %s   '"%(m,run_no,m,str(run_no)))
				
			
			elif t=='rmts' and m==95:
				os.system("sbatch -N 1 -n 8 -t 200 -o slot_attention_pretrained_heldout_unicodes_frozen_autoencoder_6slots_nowarmup_lowerlr_nolrdecay_64dim_128res_abstractor_scoring_rmts_holdout%s_run%s.log --gres=gpu:1 --constraint=gpu80 --wrap 'python -u train_slot_abstractor_rmts.py  --batch_size 16  --img_size 128 --num_epochs 400  --m_holdout %s --run %s   '"%(m,run_no,m,str(run_no)))
				
			
			

			elif t=='dist3' and m==95:
				os.system("sbatch -N 1 -n 8 -t 300 -o slot_attention_pretrained_heldout_unicodes_frozen_autoencoder_6slots_nowarmup_lowerlr_nolrdecay_64dim_128res_abstractor_scoring_dist3_holdout%s_run%s.log --gres=gpu:1 --constraint=gpu80 --wrap 'python -u train_slot_abstractor_dist3.py  --batch_size 16  --img_size 128 --num_epochs 400 --m_holdout %s --run %s  '"%(m,run_no,m,str(run_no)))
				
				
			elif t=='id' and m == 95:

				os.system("sbatch -N 1 -n 8 -t 600 -o slot_attention_pretrained_heldout_unicodes_frozen_autoencoder_6slots_nowarmup_lowerlr_nolrdecay_64dim_128res_abstractor_scoring_id_rules_holdout%s_run%s.log --gres=gpu:1 --constraint=gpu80 --wrap 'python -u train_slot_abstractor_dist3.py  --batch_size 16  --img_size 128 --num_epochs 100 --m_holdout %s --run %s --task 'identity_rules' --test_gen_method 'subsample'    '"%(m,run_no,m,str(run_no)))
				



	
