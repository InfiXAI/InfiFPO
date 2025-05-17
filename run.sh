#!/bin/bash

# Work dir
export WORK_DIR=/path/to/fpo
echo "WORK_DIR set to: $WORK_DIR"

# Conda
export CONDA_ENV=/path/to/env
export CONDA_CMD='eval "$(/path/to/conda shell.bash hook)" && conda activate ${CONDA_ENV}'
$CONDA_CMD
echo "Conda environment activated: $CONDA_DEFAULT_ENV"

# Container
export GPUS_PER_NODE=8
export CONTAINER_IMAGE=/path/to/img
export CONTAINER_NAME="fpo"
export CONTAINER_MOUNT=/path1:/path2

# Env to pass in contrainer
export WANDB_MODE=offline
export TRITON_CACHE_DIR=/path/to/cache

head_node_ip=/ip/to/head

######################

# 设置accelerate脚本
CONFIG=configs/zero3_config.yaml
SCRIPT_FILE=fpo/run.py
DATASET=/path/to/data
MODEL=/path/to/model

# 设置PO
USE_FUSION=True
USE_LN=True
FUSION_TYPE=mean

# 设置超参
MAX_PROMPT_LEN=9999
MAX_RESPONSE_LEN=9999
MAX_LEN=9999
SAVE=/path/to/output

# 动态构建路径后缀
SAVE_SUFFIX=""
if [ "$USE_FUSION" = "True" ]; then
    SAVE_SUFFIX+="-fdpo"
    if [ -n "$FUSION_TYPE" ]; then
        SAVE_SUFFIX+="-$FUSION_TYPE"
    fi
else
    SAVE_SUFFIX+="-dpo"
fi

# 添加长度归一化标识
if [ "$USE_LN" = "True" ]; then
    SAVE_SUFFIX+="-ln"
fi

# 根据USE_LN设置beta值
if [ "$USE_LN" = "True" ]; then
    BETA_VAL=2.5
    REF_OFFSET=5.0
else
    BETA_VAL=0.1
    REF_OFFSET=300
fi

export LAUNCHER="accelerate launch \
    --config_file ${WORK_DIR}/${CONFIG} \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29501 \
    --rdzv_conf rdzv_timeout=360 \
    "

export SCRIPT="${WORK_DIR}/${SCRIPT_FILE}"
export SCRIPT_ARGS=" \
    --dataset_name "$DATASET" \
    --model_name_or_path "$MODEL" \
    --learning_rate 1e-7 \
    --bf16 \
    --num_train_epochs 1 \
    --max_prompt_length $MAX_PROMPT_LEN \
    --max_completion_length $MAX_RESPONSE_LEN \
    --max_length $MAX_LEN \
    --beta $BETA_VAL \
    --ref_offset $REF_OFFSET \
    --fuse $USE_FUSION \
    --length_norm $USE_LN \
    --fusion_type $FUSION_TYPE \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --logging_first_step \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 20 \
    --save_steps 100 \
    --output_dir "$SAVE$SAVE_SUFFIX" \
    --no_remove_unused_columns
    "

# 打印完整命令用于调试
echo "Full launch command:"
echo "$LAUNCHER $SCRIPT $SCRIPT_ARGS"

# training
${LAUNCHER} ${SCRIPT} ${SCRIPT_ARGS}

echo "End of job script at $(date)"
