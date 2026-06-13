#!/bin/bash
# Training script using commonsense_170k mixed dataset (same as original HiRA setup).
# Run from the experiment/ directory.
#
# Usage:
#   bash run_train.sh                  # default: tira
#   PEFT=lora bash run_train.sh        # LoRA only (baseline)

MODEL=${MODEL:-"./models/llama-2-7b"}
OUTPUT=${OUTPUT:-"results_tira"}
PEFT=${PEFT:-"tira"}
DATASET=${DATASET:-"common_170k"}
SEED=${SEED:-"36"}
# Paper hyperparameters
EPOCH=${EPOCH:-"2"}
LR=${LR:-"2e-4"}
BATCH=${BATCH:-"32"}
GRAD_ACC=${GRAD_ACC:-"1"}
WARMUP=${WARMUP:-"100"}
TARGET_MODULES=${TARGET_MODULES:-"q_proj,k_proj,v_proj,up_proj,down_proj"}


# Eval every 80 steps, keep 20 ckpts, pick best after full training
EVAL_STRATEGY="steps"
EVAL_STEPS="80"
SAVE_TOTAL_LIMIT="10"
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
        --weight_decay 0 \
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
        --weight_decay 0 \
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
        --weight_decay 0 \
        --tira_M 32 \
        --tira_K 32 \
        --eval_strategy $EVAL_STRATEGY \
        --eval_steps $EVAL_STEPS \
        --save_total_limit $SAVE_TOTAL_LIMIT \
        --early_stop_patience $EARLY_STOP_PATIENCE \
        --output_folder $OUTPUT
fi
