#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export EXP_NAME="e10_1"

python -m training.main \
    --save-frequency 1 \
    --train-data="pipe:aws s3 cp s3://thieme/type1caption/train_{00000..00077}.tar -" \
    --train-num-samples 644278 \
    --val-data="pipe:aws s3 cp s3://thieme/type1caption/val_00000.tar -" \
    --dataset-type webdataset \
    --warmup 2000 \
    --batch-size=32 \
    --epochs=10 \
    --lr 1e-4 \
    --wd 0.2 \
    --eps 1e-6 \
    --workers=3 \
    --report-to wandb \
    --name ${EXP_NAME} \
    --logs /scratch/logs/ \
    --model "hf-hub_microsoft" \
    --coca-contrastive-loss-weight 1 \
    --coca-caption-loss-weight 2 \
    --log-every-n-steps 100 \
    --wandb-project-name stanford \
    --logs None
