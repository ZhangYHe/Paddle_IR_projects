export CUDA_VISIBLE_DEVICES=4
python train.py \
    --dataset_path /home/zhangyh/dataset/MS\ MARCO/passage\ raranking \
    --logdir /home/zhangyh/code/paddle/logs/DCE
