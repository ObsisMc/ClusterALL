# proteins
# original
python ogb_models/proteins/mlp.py --log_steps 1 --num_layers 3 --hidden_channels 256 --dropout 0.5 --lr 0.01 --epochs 1000 --eval_steps 9 --runs 10 --device 0
python ogb_models/arxiv/mlp.py --log_steps 1 --num_layers 3 --hidden_channels 256 --dropout 0.5 --lr 0.01 --epochs 1500 --runs 10 --device 0

# v_nodes
python ogb_models/proteins/main_mlp.py --log_steps 1 --num_layers 3 --hidden_channels 256 --dropout 0.5 --lr 0.01 --epochs 1000 --eval_steps 9 --runs 10 --device 0 --num_parts 5


# arxiv
python ogb_models/arxiv/main_mlp.py --log_steps 1 --num_layers 3 --hidden_channels 256 --dropout 0.5 --lr 0.01 --epochs 1500 --runs 10 --device 0 --num_parts 3 --epoch_gap 199 --dropout_cluster 0.3
