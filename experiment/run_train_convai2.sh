#!/bin/bash
# Training script for ConvAI2 dataset.
# Run from the experiment/ directory.
#
# Usage:
#   bash run_train_convai2.sh                  # default: tira
#   PEFT=lora bash run_train_convai2.sh        # LoRA only (baseline)

MODEL=${MODEL:-"./models/llama-3-8b"}
OUTPUT=${OUTPUT:-"results_tira"}
PEFT=${PEFT:-"tira"}
DATASET="convai2"
SEED=${SEED:-"36"}
# Paper hyperparameters
EPOCH=${EPOCH:-"1"}
LR=${LR:-"2e-5"}
BATCH=${BATCH:-"32"}
GRAD_ACC=${GRAD_ACC:-"1"}
WARMUP=${WARMUP:-"100"}
TARGET_MODULES=${TARGET_MODULES:-"q_proj,k_proj,v_proj,up_proj,down_proj"}
WEIGHT_DECAY=${WEIGHT_DECAY:-"0"}


# Eval every 80 steps, keep 20 ckpts, pick best after full training
EVAL_STRATEGY="steps"
EVAL_STEPS="80"
SAVE_TOTAL_LIMIT="20"
EARLY_STOP_PATIENCE="0"

echo ">>> Training on $DATASET with peft_type=$PEFT"

if [ "$PEFT" = "lora" ]; then
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --peft_type lora \
        --model $MODEL \
        --dataset $DATASET \
        --seed $SEED \
        --lr $LR \
        --batch $BATCH \
        --grad_acc $GRAD_ACC \
        --enable_grad_ckpt \
        --target_modules "$TARGET_MODULES" \
        --epoch $EPOCH \
        --warmup $WARMUP \
        --weight_decay $WEIGHT_DECAY \
        --lora_r 32 \
        --lora_alpha 32 \
        --lora_dropout 0.05 \
        --eval_strategy $EVAL_STRATEGY \
        --eval_steps $EVAL_STEPS \
        --save_total_limit $SAVE_TOTAL_LIMIT \
        --early_stop_patience $EARLY_STOP_PATIENCE \
        --output_folder $OUTPUT

elif [ "$PEFT" = "hira" ]; then
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --peft_type hira \
        --model $MODEL \
        --dataset $DATASET \
        --seed $SEED \
        --lr $LR \
        --batch $BATCH \
        --grad_acc $GRAD_ACC \
        --enable_grad_ckpt \
        --target_modules "$TARGET_MODULES" \
        --epoch $EPOCH \
        --warmup $WARMUP \
        --weight_decay $WEIGHT_DECAY \
        --r_ab 32 \
        --eval_strategy $EVAL_STRATEGY \
        --eval_steps $EVAL_STEPS \
        --save_total_limit $SAVE_TOTAL_LIMIT \
        --early_stop_patience $EARLY_STOP_PATIENCE \
        --output_folder $OUTPUT \

elif [ "$PEFT" = "tira" ]; then
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --peft_type tira \
        --model $MODEL \
        --dataset $DATASET \
        --seed $SEED \
        --lr $LR \
        --batch $BATCH \
        --grad_acc $GRAD_ACC \
        --enable_grad_ckpt \
        --target_modules "$TARGET_MODULES" \
        --epoch $EPOCH \
        --warmup $WARMUP \
        --weight_decay $WEIGHT_DECAY \
        --tira_M 32 \
        --tira_K 32 \
        --eval_strategy $EVAL_STRATEGY \
        --eval_steps $EVAL_STEPS \
        --save_total_limit $SAVE_TOTAL_LIMIT \
        --early_stop_patience $EARLY_STOP_PATIENCE \
        --output_folder $OUTPUT
fi
