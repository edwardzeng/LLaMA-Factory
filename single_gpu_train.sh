#!/usr/bin/env bash

# This is secret and shouldn't be checked into version control
export WANDB_API_KEY=$YOUR_API_KEY
# Name and notes optional
export WANDB_NAME="LLM"
export WANDB_NOTES="SFT train."


CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --report_to wandb \
    --stage sft \
    --model_name_or_path path_to_llama_model \
    --do_train \
    --dataset alpaca_gpt4_zh \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir path_to_sft_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16
