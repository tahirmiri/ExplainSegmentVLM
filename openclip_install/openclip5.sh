#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export EXP_NAME="e5_8"

python -m training.main \
    --save-frequency 1 \
    --save-most-recent \
    --train-data="/fsx/thieme/data/type1caption/train_{00000..00077}.tar" \
    --train-num-samples 644278 \
    --val-data="/fsx/thieme/data/type1caption/val_00000.tar" \
    --dataset-type webdataset \
    --warmup 2000 \
    --batch-size=256 \
    --epochs=5 \
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
    --logs None \
    --grad-checkpointing \
    --local-loss \
    --gather-with-grad 
