# proteins
# original
python ogb_models/proteins/gnn.py --log_steps 1 --num_layers 3 --hidden_channels 256 --dropout 0.5 --lr 0.01 --epochs 1000 --eval_steps 10 --runs 10 --device 0


#arxiv
python ogb_models/arxiv/main_gnn.py --log_steps 1 --num_layers 3 --hidden_channels 256 --dropout 0.5 --lr 0.01 --epochs 1000 --runs 10 --device 0 --num_parts 5
