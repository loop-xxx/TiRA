# TiRA: Tiled Rank-1 Subblock Adaptation

This repository contains the thesis source and experiment code for **TiRA: Tiled Rank-1 Subblock Adaptation for High-Rank Parameter-Efficient Fine-Tuning**.

## Repository Layout

```text
.
|-- thesis.tex                         # LaTeX source of the thesis
|-- thesis.pdf                         # Compiled thesis PDF
|-- figures/                           # Figures and figure-generation scripts
`-- experiment/
    |-- train.py                       # Main training and inference entry point
    |-- tira/                          # TiRA implementation
    |-- hira/                          # HiRA/PEFT baseline code
    |-- dataset/                       # Dataset loading and prompt formatting
    |-- customized_trainer/            # Customized Trainer
    |-- data_file.7z                   # Packed experiment data
    |-- data_file/                     # Unpacked data directory
    |-- results_tira/                  # Experiment outputs and summarized results
    `-- run_train_*.sh                 # Training scripts for different tasks
```

## Method Overview

TiRA is a parameter-efficient fine-tuning method. It keeps LoRA's bypass adapter design and mergeable-weight property, but avoids constraining the whole weight update to a globally low-rank matrix. TiRA splits the weight update matrix into an `M x M` block grid and uses `K` groups of short-vector outer products to place rank-1 sub-blocks on staggered block diagonals.

When `K = M = r`, TiRA has the same number of trainable parameters as LoRA with rank `r`, while the theoretical rank upper bound of the weight update increases from `r` to `r^2`. This is a structural rank upper bound, not a guarantee that the trained update will reach rank `r^2`. TiRA also does not represent every matrix with global rank at most `r^2`: when `K = L * M`, each block of the update has block-rank upper bound `L`. In other words, TiRA uses a local-low-rank and globally distributed block structure instead of LoRA's global low-rank factorization.

The core implementation is in `experiment/tira/layer.py`. `TiraLinear` computes the adapter output block-wise during the forward pass instead of explicitly materializing the full `Delta W`. For deployment, the adapter can be merged back into the original linear layer.

## Key Results

- On Llama-3-8B with MetaMath training and GSM8K evaluation, TiRA (`K = M = 32`) reaches `72.86%` accuracy, improving over LoRA (`r = 32`) by `6.97` percentage points and HiRA by `2.05` percentage points.
- Under the same trainable-parameter budget on Qwen-2-1.5B, LoRA (`r = 32`) obtains `68.08%` GSM8K accuracy, while TiRA with `K = 32` reaches `68.16%`, `68.39%`, and `68.84%` for `M = 2`, `M = 4`, and `M = 8`, respectively. This comparison isolates the effect of the block structure from simply increasing parameter count.
- In the `K = 32` block-size sensitivity study, the best `M` shifts upward with model size: Qwen-2-0.5B prefers `M = 2`, Qwen-2-1.5B prefers `M = 8`, and Qwen-2.5-3B prefers `M = 32`. The corresponding `KM / hidden size` ratios are `7.1%`, `16.7%`, and `50.0%`.
- Spectral analysis confirms that `KM` is only a theoretical rank upper bound. For Llama-3-8B, the measured effective rank after training is about `32` for LoRA, `432.96` for TiRA (`M = 16, K = 32`), and `757.27` for TiRA (`M = 32, K = 32`), below the corresponding theoretical upper bounds but much higher than LoRA.

## Environment

Create the Conda environment from the provided file:

```bash
cd experiment
conda env create -f env.yml
conda activate tira
```

## Data

The experiment datasets is packed as:

```text
experiment/data_file.7z
```

The code reads datasets from `data_file/` by default. The main subdirectories are:

- `data_file/llm_adapt/`: commonsense reasoning and GSM8K-related data
- `data_file/convai2/`: ConvAI2 dialogue-generation data
- `data_file/meta_math/`: MetaMath training data

## Models

The training code can load either Hugging Face model IDs or local model paths. The experiment scripts assume local paths such as:

```text
experiment/models/llama-2-7b
experiment/models/llama-3-8b
experiment/models/qwen-2-1.5b
experiment/models/qwen-2.5-3b
```

If your models are stored elsewhere, pass the path or Hugging Face model ID with `--model_name`.

## Training

Run commands from the `experiment/` directory.

Example TiRA training command on MetaMath:

```bash
python train.py \
  --peft_type tira \
  --model_name ./models/llama-3-8b \
  --dataset meta_math \
  --seed 36 \
  --lr 4e-5 \
  --batch 32 \
  --grad_acc 1 \
  --enable_grad_ckpt \
  --target_modules "q_proj,k_proj,v_proj,up_proj,down_proj" \
  --epoch 2 \
  --warmup 100 \
  --weight_decay 0 \
  --tira_M 32 \
  --tira_K 32 \
  --eval_strategy steps \
  --eval_steps 80 \
  --save_total_limit 5 \
  --output_folder results_tira
```

Example LoRA baseline command:

```bash
python train.py \
  --peft_type lora \
  --model_name ./models/llama-3-8b \
  --dataset meta_math \
  --seed 36 \
  --lr 4e-5 \
  --batch 32 \
  --grad_acc 1 \
  --enable_grad_ckpt \
  --target_modules "q_proj,k_proj,v_proj,up_proj,down_proj" \
  --epoch 2 \
  --warmup 100 \
  --weight_decay 0 \
  --lora_r 32 \
  --lora_alpha 32 \
  --lora_dropout 0 \
  --eval_strategy steps \
  --eval_steps 80 \
  --save_total_limit 5 \
  --output_folder results_tira
```

The `run_train_*.sh` scripts provide task-specific presets. You can override model path, output folder, PEFT type, and other settings through environment variables:

```bash
MODEL=./models/llama-3-8b PEFT=tira bash run_train_metamath.sh
```

## Evaluation and Results

When `--ckpt` is passed to `train.py`, the script runs inference and writes jsonl outputs under the checkpoint directory. Result aggregation scripts include:

```text
eval_commonsense.py
convai2_eval.py
metamath_eval.py
eval_gsm8k.py
```

These evaluation scripts contain `<fill your result dir here>` placeholders. Replace them with the actual result directory before re-running evaluation.

Existing summarized results are included in:

- `results_tira/results_gsm8k-easy.csv`: GSM8K Easy motivation experiment
- `results_tira/results_gsm8k-challenge.csv`: GSM8K Challenge motivation experiment
- `results_tira/results_meta_math.csv`: MetaMath training and GSM8K evaluation results
- `results_tira/results_ablation_meta_math.csv`: TiRA block-size `M` sensitivity analysis results
- `results_tira/llama-*-commonsense/`: eight-task commonsense reasoning results
- `results_tira/llama-*-convai2/`: ConvAI2 dialogue-generation results

## Key Files

Useful files for understanding the implementation:

- `experiment/tira/layer.py`: TiRA layer and block-wise forward computation
- `experiment/tira/model.py`: target linear-layer replacement, adapter management, and weight merging
- `experiment/tira/config.py`: TiRA configuration
- `experiment/models/get_models.py`: model loading and PEFT method selection
- `experiment/train.py`: training, inference, and output-saving workflow
