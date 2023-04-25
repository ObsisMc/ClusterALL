# proteins
# original
python ogb_models/proteins/mlp.py --log_steps 1 --num_layers 3 --hidden_channels 256 --dropout 0.5 --lr 0.01 --epochs 1000 --eval_steps 10 --runs 5 --device 0

# v_nodes
python ogb_models/proteins/main_mlp.py --log_steps 1 --num_layers 3 --hidden_channels 256 --dropout 0.5 --lr 0.01 --epochs 1000 --eval_steps 10 --runs 5 --device 0 --num_parts 5


# arxiv
python ogb_models/arxiv/main_mlp.py --log_steps 1 --num_layers 3 --hidden_channels 256 --dropout 0.5 --lr 0.01 --epochs 1000 --runs 10 --device 0 --num_parts 5
