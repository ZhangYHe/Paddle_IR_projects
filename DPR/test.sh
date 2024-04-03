export CUDA_VISIBLE_DEVICES=5
python test.py \
    --file_path /home/zhangyh/code/paddle/DPR_paddle/data/SQuAD2.0/train-v2.0.json \
    --checkpoint_path /home/zhangyh/code/paddle/DPR_paddle/checkpoints/checkpointsdpr_model_epoch_2.pdparams
