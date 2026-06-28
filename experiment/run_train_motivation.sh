#!/bin/bash

MODEL=${MODEL:-"./models/qwen-2-1.5b"}
OUTPUT=${OUTPUT:-"results_tira"}
PEFT=${PEFT:-"$2"}
DATASET=${DATASET:-"gsm8k-$1"}
SEED=${SEED:-"36"}
# Paper hyperparameters
EPOCH=${EPOCH:-"2"}
LR=${LR:-"$3"}
BATCH=${BATCH:-"160"}
GRAD_ACC=${GRAD_ACC:-"1"}
WARMUP=${WARMUP:-"100"}
TARGET_MODULES=${TARGET_MODULES:-"q_proj,k_proj,v_proj,up_proj,down_proj"}


# Eval every 80 steps, keep 20 ckpts, pick best after full training
EVAL_STRATEGY="steps"
EVAL_STEPS="2"
SAVE_TOTAL_LIMIT="1"
EARLY_STOP_PATIENCE="0"

echo ">>> Training on $DATASET with peft_type=$PEFT"

if [ "$PEFT" = "lora" ]; then
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --peft_type lora \
        --model_name $MODEL \
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
        --lora_r $4 \
        --lora_alpha $4 \
        --lora_dropout 0.05 \
        --eval_strategy $EVAL_STRATEGY \
        --eval_steps $EVAL_STEPS \
        --save_total_limit $SAVE_TOTAL_LIMIT \
        --early_stop_patience $EARLY_STOP_PATIENCE \
        --output_folder $OUTPUT

elif [ "$PEFT" = "hira" ]; then
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --peft_type hira \
        --model_name $MODEL \
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
        --model_name $MODEL \
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
        --tira_M $5 \
        --tira_K $6 \
        --eval_strategy $EVAL_STRATEGY \
        --eval_steps $EVAL_STEPS \
        --save_total_limit $SAVE_TOTAL_LIMIT \
        --early_stop_patience $EARLY_STOP_PATIENCE \
        --output_folder $OUTPUT
fi
