#!/bin/bash
sudo apt update -y && \
sudo apt install libglu1-mesa-dev -y && \NCCL_P2P_DISABLE="1"
pip install poetry "huggingface_hub[cli]" && \
huggingface-cli login --token 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 && \
mkdir -p content && mkdir -p content/vivid-llama2-7b && \
huggingface-cli download Vividbot/llava-pretrain-vi --repo-type dataset --include llava_pretrain_vi_all.json --local-dir ./content && \
huggingface-cli download Vividbot/vast-2m-vi --repo-type dataset --include vast_2m_vi_refined_all.json --local-dir ./content && \
git clone https://dminhvu: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@github.com/FPT-VVU/ViVidBot && \
cd ViVidBot && git checkout train/llamavid && \
poetry install && poetry add deepspeed wandb && \
poetry run pip install flash-attn -U --force-reinstall && \
wandb login c2842eff34b9959f6e3efe3a790707d7ccf10fb3

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed vividbot/llamavid/train/train_mem.py \
    --deepspeed ./vividbot/scripts/zero2.json \
    --model_name_or_path meta-llama/Llama-2-7b \
    --version plain_guided \
    --data_path /content/vast_2m_vi_refined_all.json /content/llava_pretrain_vi_all.json \
    --image_folder Vividbot/llava-pretrain-vi/images \
    --video_folder Vividbot/vast-2m-vi/video \
    --vision_tower openai/clip-vit-large-patch14 \
    --image_processor ./vividbot/llamavid/processor/clip-patch14-224 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --video_fps 1 \
    --bert_type "qformer_pretrain_freeze" \
    --num_query 32 \
    --compress_type "mean" \
    --bf16 True \
    --output_dir /content/vivid-llama2-7b \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

deepspeed llamavid/train/train_mem.py \
    --deepspeed ./scripts/zero2_offload.json \
    --model_name_or_path model_zoo/LLM/vicuna/7B-V1.5 \
    --version imgsp_v1 \
    --data_path ./data/LLaMA-VID-Finetune/llava_v1_5_mix665k_with_video_chatgpt.json \
    --image_folder ./data/LLaMA-VID-Finetune \
    --video_folder ./data/LLaMA-VID-Finetune \
    --vision_tower ./model_zoo/LAVIS/eva_vit_g.pth \
    --image_processor ./llamavid/processor/clip-patch14-224 \
    --pretrain_mm_mlp_adapter ./work_dirs/llama-vid-7b-pretrain-224-video-fps-1/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --video_fps 1 \
    --bert_type "qformer_pretrain" \
    --num_query 32 \
    --compress_type "mean" \
    --bf16 True \
    --output_dir ./work_dirs/llama-vid-7b-full-224-video-fps-1  \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
