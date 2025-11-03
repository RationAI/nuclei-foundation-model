#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4            # Request N GPUs per machine
#SBATCH --mem=0
#SBATCH -D /project/project_465002057/nuclei-foundational-model
#SBATCH --time=0-12:00:00
#SBATCH --account=project_465002057
#SBATCH --partition=standard-g

export MLFLOW_TRACKING_URI="file:///scratch/project_465002057/mlruns"

srun singularity exec \
    -B "/flash/project_465002057,/scratch/project_465002057,/projappl/project_465002057" \
    "/project/project_465002057/nfm_0.4.0-rocm.sif" \
    python3 -m nfm mode=fit +experiment=LUMI +trainer.num_nodes=$SLURM_NNODES +trainer.devices=$SLURM_GPUS_ON_NODE +trainer.strategy=deepspeed_stage_2
