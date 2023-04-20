# proteins
# original
python ogb/mlp.py --log_steps 1 --num_layers 3 --hidden_channels 256 --dropout 0.5 --lr 0.01 --epochs 1000 --eval_steps 10 --runs 5 --device 0

# k v_nodes
python ogb/main_mlp.py --log_steps 1 --num_layers 3 --hidden_channels 256 --dropout 0.5 --lr 0.01 --epochs 1000 --eval_steps 10 --runs 5 --device 0 --num_parts 5

