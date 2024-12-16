accelerate launch --mixed_precision="bf16" overfit_batch.py \
    --data_path=data.json \
    --video_dir="/workspace/OpenVid_part1/" \
    --checkpointing_steps 50 \
    --train_batch_size 2