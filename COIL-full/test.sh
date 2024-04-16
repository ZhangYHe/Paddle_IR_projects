export CUDA_VISIBLE_DEVICES=[0
python test.py \
    --dataset_path /home/zhangyh/dataset/MS\ MARCO/passage\ raranking \
    --checkpoint_path /home/zhangyh/code/paddle/COIL-full_paddle/checkpoints/COIL-full_checkpoint_3000.pdparams \
    --logdir /home/zhangyh/code/paddle/logs/COIL-full-test
