# Please run in the root directory

# proteins
python nodeformer/main_nodeformer.py --dataset ogbn-proteins --metric rocauc --method nodeformer_encoder --lr 1e-2 --weight_decay 0. --num_layers 3 --hidden_channels 64 --num_heads 1 --rb_order 1 --rb_trans identity --lamda 0.1 --M 50 --K 5 --use_bn --use_residual --use_gumbel --use_act --use_jk --batch_size 5000 --runs 5 --epochs 1500 --eval_step 9 --device 0 --num_parts 5 --shuffle  --save_model

# arxiv
python nodeformer/main_nodeformer.py --dataset ogbn-arxiv --metric acc --method nodeformer_encoder --lr 1e-2 --weight_decay 0. --num_layers 3 --hidden_channels 64 --num_heads 1 --rb_order 1 --rb_trans identity --lamda 0.1 --M 50 --K 5 --use_bn --use_residual --use_gumbel --use_act --use_jk --batch_size 10000 --runs 5 --epochs 1000 --eval_step 9 --device 0 --num_parts 50 --shuffle  --save_model
