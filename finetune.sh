accelerate launch --mixed_precision="bf16" finetune.py \
    --data_path=data.json \
    --video_dir="/workspace/LTX-Video/OpenVid_part1/" \
    --checkpointing_steps 2