# original
## ogbn-proteins
python ogb_models/proteins/gnn.py --log_steps 1 --num_layers 3 --hidden_channels 256 --dropout 0.5 --lr 0.01 --epochs 1000 --eval_steps 10 --runs 5 --device 0 --use_sage
## ogbn-arxiv
python ogb_models/arxiv/gnn.py --log_steps 1 --num_layers 3 --hidden_channels 256 --dropout 0.5 --lr 0.01 --epochs 1000 --runs 10 --device 0 --use_sage


# ClusterALL
## ogbn-proteins
python ogb_models/proteins/main_gnn.py --log_steps 1 --num_layers 3 --hidden_channels 256 --dropout 0.5 --lr 0.01 --epochs 1500 --eval_steps 9 --runs 5 --device 0 --num_parts 2 --epoch_gap 199 --dropout_cluster 0 --warm_up 0 --use_sage
## ogbn-arxiv
python ogb_models/arxiv/main_gnn.py --log_steps 1 --num_layers 3 --hidden_channels 256 --dropout 0.5 --lr 0.01 --epochs 1500 --runs 5 --device 0 --num_parts 5 --epoch_gap 199 --use_sage  --dropout_cluster 0.3
