#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4            # Request N GPUs per machine
#SBATCH --mem=0
#SBATCH --time=0-02:00:00

# Activate conda environment
source activate $1

srun uv run python -m nfm +trainer.num_nodes=$SLURM_NNODES +trainer.devices=$SLURM_GPUS_PER_NODE +trainer.strategy=deepspeed_stage_2