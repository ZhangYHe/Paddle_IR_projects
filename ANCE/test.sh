export CUDA_VISIBLE_DEVICES=3
python test.py \
    --dataset_path /home/zhangyh/dataset/MS\ MARCO/passage\ raranking \
    --checkpoint_path /home/zhangyh/code/paddle/ANCE_paddle/checkpoints/ANCE_checkpoint_500.pdparams \
    --logdir /home/zhangyh/code/paddle/logs/ANCE-test
