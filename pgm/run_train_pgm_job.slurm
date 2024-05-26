#!/bin/bash


#SBATCH --nodes=1               # node count
#SBATCH --ntasks-per-node=4      # total number of tasks per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --constraint=gpu80
#SBATCH --time=144:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=slot_attention_pretrained_neutral_train_frozen_autoencoder_abstractor_scoring_multigpu_tcn_16slots_dspritesdecoder_lowerlr_rowcolposemb_added_its=3_extrapolation_pgm_run1.log
source ~/.bashrc
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export NCCL_DEBUG=INFO


srun python train_slot_abstractor_pgm.py --img_size=80 --batch_size=24 --depth=24 --learning_rate=0.00008 --run='1' --num_epochs=100 --path='pgm_datasets/extrapolation/'