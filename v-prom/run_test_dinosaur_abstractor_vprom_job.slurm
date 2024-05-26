#!/bin/bash


#SBATCH --nodes=1               # node count
#SBATCH --ntasks-per-node=4      # total number of tasks per node
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=32G 
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=dinosaur_vitb16_pretrained_slotattn_mlpdecoder_abstractor_multigpu_10weightmse_assymm_qkweights_initsame_layerwise_original_vitposcodes_0.1dropout_tcn_11slots_lowerlr_nolrdecay_rowcolposemb_added_vprom_neutral_test.log
source ~/.bashrc
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export NCCL_DEBUG=INFO


srun python test_dinosaur_abstractor_vprom.py  --batch_size=8 --depth=24  