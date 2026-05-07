#!/bin/bash
# train_dna_validation_cluster.sh
#
# Trains independent validation CNN regressors (DNARegressorV2) for reward-hacking ablation.
# One job per task. Results saved to: cnn_dna_models_v2/{task}/
#
# Usage: bash Big_Oracles/DNA/train_dna_validation_cluster.sh

set -eo pipefail

BASE=/scratch/gpfs/MONA/Toki/GRACE/protein_grace
cd "$BASE"
mkdir -p logs

SCRIPT=${BASE}/Big_Oracles/DNA/train_dna_validation.py

for TASK in hepG2 k562 sknsh; do
    JOB="dna_val_${TASK}"

    sbatch \
        --job-name="$JOB" \
        --output="logs/${JOB}_%j.out" \
        --error="logs/${JOB}_%j.err" \
        --gres=gpu:1 \
        --mem=32G \
        --cpus-per-task=4 \
        --time=00:59:00 \
        --mail-type=FAIL \
        --mail-user=mt3204@princeton.edu \
        --wrap="
            set -eo pipefail
            module purge
            module load anaconda3/2024.2
            source \"\$(conda info --base)/etc/profile.d/conda.sh\"
            conda activate tftrain
            cd ${BASE}/Big_Oracles/DNA
            export TRANSFORMERS_OFFLINE=1
            export HF_DATASETS_OFFLINE=1

            echo 'Task  : ${TASK}'
            echo 'Node  : '\$(hostname)
            echo 'Start : '\$(date)

            python ${SCRIPT} \
                --task        ${TASK} \
                --epochs      9 \
                --batch_size  128 \
                --lr          5e-4 \
                --weight_decay 1e-4 \
                --num_workers 4 \
                --seed        99
        "

    echo "Submitted: ${JOB}"
done
