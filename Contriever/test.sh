export CUDA_VISIBLE_DEVICES=4
python test.py \
    --dataset_path /home/zhangyh/dataset/MS\ MARCO/passage\ raranking \
    --checkpoint_path /home/zhangyh/code/paddle/Contriever_paddle/checkpoints/Contriever_model_checkpoint_4000.pdparams
