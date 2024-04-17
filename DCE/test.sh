export CUDA_VISIBLE_DEVICES=4
python test.py \
    --dataset_path /home/zhangyh/dataset/MS\ MARCO/passage\ raranking \
    --checkpoint_path /home/zhangyh/code/paddle/DCE_paddle/checkpoints/DCE_checkpoint_4500.pdparams \
    --logdir /home/zhangyh/code/paddle/logs/DCE-test
