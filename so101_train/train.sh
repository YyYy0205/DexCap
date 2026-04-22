#!/usr/bin/env bash
# SO-101 双臂 imitation learning 训练脚本
#
# 用法：
#   cd so101_train
#   bash train.sh          # BC-MLP（快，适合验证流程）
#   bash train.sh rnn      # BC-RNN（推荐，适合连续操作任务）
#   bash train.sh --epochs 200 --batch 32   # 覆盖超参数
#
# 依赖：
#   conda activate lerobot
#   pip install robomimic --no-deps   # 如未安装

set -e

ALGO="${1:-bc}"
CONFIG="bc_so101.json"

if [ "$ALGO" = "rnn" ]; then
    CONFIG="bc_rnn_so101.json"
fi

# 允许覆盖 epochs/batch
EXTRA_ARGS=""
for arg in "$@"; do
    case "$arg" in
        --epochs=*) EXTRA_ARGS="$EXTRA_ARGS train.num_epochs=${arg#*=}" ;;
        --batch=*)  EXTRA_ARGS="$EXTRA_ARGS train.batch_size=${arg#*=}" ;;
    esac
done

echo "================================================"
echo "Training config : $CONFIG"
echo "Extra overrides : $EXTRA_ARGS"
echo "================================================"

python -m robomimic.scripts.train \
    --config "$CONFIG" \
    $EXTRA_ARGS

echo ""
echo "Done. Checkpoints saved in trained_models/"
