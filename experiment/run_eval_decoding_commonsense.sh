#!/bin/bash

MODEL=${MODEL:-"./models/$1"}
BATCH=${BATCH:-"32"}

CUDA_VISIBLE_DEVICES=0 python train.py --dataset=boolq --batch=$BATCH --output_folder=temp --ckpt=$2 --beam_size=1 \
    --peft_type=tira \
    --model $MODEL \
    --seed 36 \
    --target_modules "q_proj,k_proj,v_proj,up_proj,down_proj" \
    --tira_M 32 \
    --tira_K 32 
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=piqa --batch=$BATCH --output_folder=temp --ckpt=$2 --beam_size=1 \
    --peft_type=tira \
    --model $MODEL \
    --seed 36 \
    --target_modules "q_proj,k_proj,v_proj,up_proj,down_proj" \
    --tira_M 32 \
    --tira_K 32 
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=siqa --batch=$BATCH --output_folder=temp --ckpt=$2 --beam_size=1 \
    --peft_type=tira \
    --model $MODEL \
    --seed 36 \
    --target_modules "q_proj,k_proj,v_proj,up_proj,down_proj" \
    --tira_M 32 \
    --tira_K 32 
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=arcc --batch=$BATCH --output_folder=temp --ckpt=$2 --beam_size=1 \
    --peft_type=tira \
    --model $MODEL \
    --seed 36 \
    --target_modules "q_proj,k_proj,v_proj,up_proj,down_proj" \
    --tira_M 32 \
    --tira_K 32 
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=arce --batch=$BATCH --output_folder=temp --ckpt=$2 --beam_size=1 \
    --peft_type=tira \
    --model $MODEL \
    --seed 36 \
    --target_modules "q_proj,k_proj,v_proj,up_proj,down_proj" \
    --tira_M 32 \
    --tira_K 32 

CUDA_VISIBLE_DEVICES=0 python train.py --dataset=obqa --batch=$BATCH --output_folder=temp --ckpt=$2 --beam_size=1 \
    --peft_type=tira \
    --model $MODEL \
    --seed 36 \
    --target_modules "q_proj,k_proj,v_proj,up_proj,down_proj" \
    --tira_M 32 \
    --tira_K 32 
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=hellas --batch=$BATCH --output_folder=temp --ckpt=$2 --beam_size=1 \
    --peft_type=tira \
    --model $MODEL \
    --seed 36 \
    --target_modules "q_proj,k_proj,v_proj,up_proj,down_proj" \
    --tira_M 32 \
    --tira_K 32 
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=winog --batch=$BATCH --output_folder=temp --ckpt=$2 --beam_size=1 \
    --peft_type=tira \
    --model $MODEL \
    --seed 36 \
    --target_modules "q_proj,k_proj,v_proj,up_proj,down_proj" \
    --tira_M 32 \
    --tira_K 32 