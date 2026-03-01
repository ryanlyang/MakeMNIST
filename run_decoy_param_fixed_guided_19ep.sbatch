#!/bin/bash -l
# Fixed guided DecoyMNIST run:
# - keep Decoy training protocol
# - transfer guided method knobs from WB100
# - select checkpoint by best val_acc after attention epoch

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/decoy_fixed_guided_19ep_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/decoy_fixed_guided_19ep_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

REPO_ROOT="/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST"
CONDA_ENV="learntolook"
LOG_DIR="/home/ryreu/guided_cnn/logsMNIST"
GT_PATH="/home/ryreu/guided_cnn/MNIST_AGAIN/DecoyGen/LearningToLook/code/WeCLIPPlus/results_decoy_mnist/val/prediction_cmap"

# Fixed run settings
EPOCHS=19
ATTN_SRC=73
SRC_EPOCHS=200
# attention epoch transfer by normalized epoch fraction: round((ATTN_SRC / SRC_EPOCHS) * EPOCHS)
ATTENTION_EPOCH=7
KL_LAMBDA=495.60509512105125
KL_INCR=0.0
LR2_MULT=1.0

# Match the CDEP-style vanilla setup: same Adam LR for backbone/classifier.
BASE_LR=0.031210590691245817
CLASSIFIER_LR=0.031210590691245817

RUN_TAG="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-manual}"
OPTUNA_SEED="$(date +%s)"
OBJECTIVE_SEED="$((OPTUNA_SEED % 100000 + 1))"
STUDY_NAME="decoy_fixed_guided_19ep_${RUN_TAG}"
DB_PATH="${REPO_ROOT}/models/DecoyMNIST/${STUDY_NAME}.db"

mkdir -p "${LOG_DIR}"
mkdir -p "${REPO_ROOT}/models/DecoyMNIST"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

cd "${REPO_ROOT}"

echo "[$(date)] Host: $(hostname)"
which python
echo "Study: ${STUDY_NAME}"
echo "Optuna seed: ${OPTUNA_SEED}"
echo "Objective seed: ${OBJECTIVE_SEED}"
echo "Epoch mapping: ${ATTN_SRC}/${SRC_EPOCHS} -> attention_epoch=${ATTENTION_EPOCH} for epochs=${EPOCHS}"

python -u mnist/run_decoy_param_optuna.py \
  --gt-path "${GT_PATH}" \
  --epochs "${EPOCHS}" \
  --val-frac 0.1 \
  --batch-size 64 \
  --test-batch-size 1000 \
  --weight-decay 1e-4 \
  --beta 10 \
  --val-saliency igrad \
  --val-metric fwd_kl \
  --checkpoint-selector val_acc \
  --selector-start attention \
  --n-trials 1 \
  --n-seeds 5 \
  --objective-seed "${OBJECTIVE_SEED}" \
  --optuna-seed "${OPTUNA_SEED}" \
  --attention-epoch-min "${ATTENTION_EPOCH}" \
  --attention-epoch-max "${ATTENTION_EPOCH}" \
  --kl-lambda-min "${KL_LAMBDA}" \
  --kl-lambda-max "${KL_LAMBDA}" \
  --kl-incr-fixed "${KL_INCR}" \
  --lr-min "${BASE_LR}" \
  --lr-max "${BASE_LR}" \
  --lr2-min "${CLASSIFIER_LR}" \
  --lr2-max "${CLASSIFIER_LR}" \
  --lr2-mult-min "${LR2_MULT}" \
  --lr2-mult-max "${LR2_MULT}" \
  --study-name "${STUDY_NAME}" \
  --db-path "${DB_PATH}"
