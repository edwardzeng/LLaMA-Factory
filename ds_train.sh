#!/usr/bin/env bash

export WANDB_PROJECT=LLM
# Log histograms of gradients and parameters
export WANDB_WATCH=all


deepspeed --include localhost:1,2,3,4 --master_port=9901 src/train_bash.py \
    --deepspeed ds_config.json \
    --report_to wandb \
    --stage sft \
    --model_name_or_path path_to_llama_model \
    --do_train \
    --dataset alpaca_gpt4_zh \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir ./output_sft \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 6.0 \
    --plot_loss \
    --fp16
