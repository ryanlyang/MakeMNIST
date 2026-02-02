#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/optuna_decoymnist_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/optuna_decoymnist_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail
mkdir -p /home/ryreu/guided_cnn/logsMNIST

REPO_ROOT="/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST"
CONDA_ENV="learntolook"

DATA_PATH="/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png"
GT_PATH="${GT_PATH:-/home/ryreu/guided_cnn/MNIST_AGAIN/DecoyGen/LearningToLook/code/WeCLIPPlus/results_decoy_mnist/val/prediction_cmap}"
STUDY_NAME="${STUDY_NAME:-decoymnist_guided}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

cd "${REPO_ROOT}"

echo "[$(date)] Host: $(hostname)"
which python

echo "Running Optuna sweep for guided LeNet on DecoyMNIST"
echo "Data: ${DATA_PATH}"
echo "Masks: ${GT_PATH}"

srun --unbuffered python -u mnist/run_decoy_optuna.py \
  --png-root "${DATA_PATH}" \
  --gt-path "${GT_PATH}" \
  --study-name "${STUDY_NAME}" \
  --epochs 30 \
  --sweep-hours 24 \
  --n-seeds 10
