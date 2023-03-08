#!/bin/bash
mkdir -p model
mkdir -p log
INPUT_DIR=`readlink -e samples`
OUTPUT_DIR=`readlink -e model`
LOG_DIR=`readlink -e log`
BASE_MODEL=`readlink -e ../chilloutmix_NiPrunedFp32Fix.safetensors`
cd ../kohya_ss
accelerate launch \
--num_cpu_threads_per_process=2 \
"train_network.py" \
--pretrained_model_name_or_path="$BASE_MODEL" \
--train_data_dir=$INPUT_DIR \
--resolution=512,512 \
--output_dir=$OUTPUT_DIR \
--logging_dir=$LOG_DIR \
--network_alpha="128" \
--save_model_as=safetensors \
--network_module=networks.lora \
--text_encoder_lr=5e-5 \
--unet_lr=0.0001 \
--network_dim=128 \
--output_name="lora" \
--lr_scheduler_num_cycles="1" \
--learning_rate="0.0001" \
--lr_scheduler="constant" \
--train_batch_size="1" \
--save_every_n_epochs="1" \
--mixed_precision="fp16" \
--save_precision="fp16" \
--seed="1234" \
--caption_extension=".txt" \
--cache_latents \
--optimizer_type="AdamW" \
--max_data_loader_n_workers="1" \
--clip_skip=2 \
--bucket_reso_steps=64 \
--mem_eff_attn \
--gradient_checkpointing \
--xformers \
--bucket_no_upscale



