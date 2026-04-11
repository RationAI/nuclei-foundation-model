#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8       # Request N GPUs per machine
#SBATCH --mem=0                  # Request all memory
#SBATCH -D /project/project_465002057/nuclei-foundational-model
#SBATCH --time=0-24:00:00
#SBATCH --account=project_465002057
#SBATCH --partition=standard-g

CPU_BIND="mask_cpu:7e000000000000,7e00000000000000"
CPU_BIND="${CPU_BIND},7e0000,7e000000"
CPU_BIND="${CPU_BIND},7e,7e00"
CPU_BIND="${CPU_BIND},7e00000000,7e0000000000"

export MLFLOW_TRACKING_URI="file:///scratch/project_465002057/mlruns"
export MPICH_GPU_SUPPORT_ENABLED=1

srun --cpu-bind=${CPU_BIND} singularity exec \
    -B "/flash/project_465002057,/scratch/project_465002057,/projappl/project_465002057" \
    "/project/project_465002057/nfm_0.6.0-rocm.sif" \
    python3 -m nfm mode=fit +experiment=LUMI +trainer.num_nodes=$SLURM_NNODES +trainer.devices=$SLURM_GPUS_ON_NODE \
        +trainer.strategy._target_=lightning.pytorch.strategies.DeepSpeedStrategy \
        +trainer.strategy.stage=2 \
        +trainer.strategy.config.train_micro_batch_size_per_gpu=32
