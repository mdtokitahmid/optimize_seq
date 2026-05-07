#!/bin/bash
# =============================================================================
# train_dna_cluster.sh
# Submit one separate cluster job per DNA task by looping over tasks locally.
#
# Submit:
#   bash Big_Oracles/DNA/train_dna_cluster.sh
#
# This is a lightweight launcher job.
# It issues 3 separate sbatch submissions:
#   - hepG2
#   - k562
#   - sknsh
# =============================================================================


set -eo pipefail

BASE=/scratch/gpfs/MONA/Toki/GRACE/protein_grace
SCRIPT=${BASE}/Big_Oracles/DNA/train_cnn_dna.py

cd "$BASE"
mkdir -p logs

for TASK in hepG2 k562 sknsh; do
    JOB="dna_${TASK}"

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
            cd ${BASE}
            export TRANSFORMERS_OFFLINE=1
            export HF_DATASETS_OFFLINE=1

            echo 'Task  : ${TASK}'
            echo 'Node  : '\$(hostname)
            echo 'Start : '\$(date)

            python ${SCRIPT} \
                --task ${TASK} \
                --epochs 20 \
                --batch_size 128 \
                --lr 1e-3 \
                --weight_decay 1e-4 \
                --num_workers 4 \
                --seed 42
        "

    echo "Submitted ${JOB}"
done
