export CUDA_VISIBLE_DEVICES=0
python test.py \
    --dataset_path /home/zhangyh/dataset/MS\ MARCO/passage\ raranking \
    --checkpoint_path /home/zhangyh/code/paddle/ColBERT_paddle/checkpoints/colbert_checkpoint_1500.pdparams
