#!/bin/bash

MODEL=${MODEL:-"./models/qwen-2.5-3b"}
OUTPUT=${OUTPUT:-"results_tira"}
PEFT=${PEFT:-"$1"}
DATASET=${DATASET:-"meta_math"}
SEED=${SEED:-"36"}
# Paper hyperparameters
EPOCH=${EPOCH:-"2"}
LR=${LR:-"5e-5"}
BATCH=${BATCH:-"32"}
GRAD_ACC=${GRAD_ACC:-"1"}
WARMUP=${WARMUP:-"100"}
TARGET_MODULES=${TARGET_MODULES:-"q_proj,k_proj,v_proj,up_proj,down_proj"}


# Eval every 100 steps, keep 5 ckpts, pick best after full training
EVAL_STRATEGY="steps"
EVAL_STEPS="100"
SAVE_TOTAL_LIMIT="5"
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
        --lora_r 32 \
        --lora_alpha 32 \
        --lora_dropout 0 \
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
    
elif [ "$PEFT" = "tira" ] || [ "$PEFT" = "tira-diagonal" ] || [ "$PEFT" = "tira-upper-triangular" ] || [ "$PEFT" = "tira-lower-triangular" ]; then
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --peft_type $PEFT \
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
        --tira_M 32 \
        --tira_L $2 \
        --eval_strategy $EVAL_STRATEGY \
        --eval_steps $EVAL_STEPS \
        --save_total_limit $SAVE_TOTAL_LIMIT \
        --early_stop_patience $EARLY_STOP_PATIENCE \
        --output_folder $OUTPUT
fi
