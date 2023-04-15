import os
import time
import torch
import torch.nn as nn


def save_ckpt(model: nn.Module, args, add_config: dict = None):
    model_name = args.method
    save_dir = os.path.join(args.model_dir, f'{args.dataset}/{model_name}/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ckpt_name = ''
    if model_name == "nodeformer":
        ckpt_name = f'bz{args.batch_size}_M{args.M}_K{args.K}_nl{args.num_layers}_hc{args.hidden_channels}_' \
                    f'nh{args.num_heads}_lmd{args.lamda}_ro{args.rb_order}_ug{args.use_gumbel}_ub{args.use_bn}_' \
                    f'ur{args.use_residual}_ua{args.use_act}_uj{args.use_jk}_rt{args.rb_trans}_lr{args.lr}_' \
                    f'wd{args.weight_decay}_'
    elif model_name == "nodeformer_encoder":
        ckpt_name = f'bz{args.batch_size}_M{args.M}_K{args.K}_nl{args.num_layers}_hc{args.hidden_channels}_' \
                    f'nh{args.num_heads}_lmd{args.lamda}_ro{args.rb_order}_ug{args.use_gumbel}_ub{args.use_bn}_' \
                    f'ur{args.use_residual}_ua{args.use_act}_uj{args.use_jk}_rt{args.rb_trans}_lr{args.lr}_' \
                    f'wd{args.weight_decay}_np{args.num_parts}_'
    elif model_name == "nodeformer_encoderfc":
        ckpt_name = f'bz{args.batch_size}_M{args.M}_K{args.K}_nl{args.num_layers}_hc{args.hidden_channels}_' \
                    f'nh{args.num_heads}_lmd{args.lamda}_ro{args.rb_order}_ug{args.use_gumbel}_ub{args.use_bn}_' \
                    f'ur{args.use_residual}_ua{args.use_act}_uj{args.use_jk}_rt{args.rb_trans}_lr{args.lr}_' \
                    f'wd{args.weight_decay}_np{args.num_parts}_wue{args.warmup_epoch}_'

    if add_config is not None:
        for key, item in add_config.items():
            ckpt_name += f'{key}{item}_'

    ckpt_name = ckpt_name.rstrip('_')
    ckpt_name += '.pkl'
    torch.save(model.state_dict(), os.path.join(save_dir, ckpt_name))
    print(f"Save ckpt into {save_dir}/{ckpt_name}")


def load_ckpt(model: nn.Module, args, device, custom_name: str = None):
    save_dir = os.path.join(args.model_dir, f'{args.dataset}/{args.method}/')
    ckpt_name = ''
    if custom_name is not None:
        ckpt_name = custom_name
    elif args.method == "nodeformer":
        ckpt_name = f'bz{args.batch_size}_M{args.M}_K{args.K}_nl{args.num_layers}_hc{args.hidden_channels}_' \
                    f'nh{args.num_heads}_lmd{args.lamda}_ro{args.rb_order}_ug{args.use_gumbel}_ub{args.use_bn}_' \
                    f'ur{args.use_residual}_ua{args.use_act}_uj{args.use_jk}_rt{args.rb_trans}_lr{args.lr}_' \
                    f'wd{args.weight_decay}'
    elif args.method == "nodeformer_encoder":
        ckpt_name = f'bz{args.batch_size}_M{args.M}_K{args.K}_nl{args.num_layers}_hc{args.hidden_channels}_' \
                    f'nh{args.num_heads}_lmd{args.lamda}_ro{args.rb_order}_ug{args.use_gumbel}_ub{args.use_bn}_' \
                    f'ur{args.use_residual}_ua{args.use_act}_uj{args.use_jk}_rt{args.rb_trans}_lr{args.lr}_' \
                    f'wd{args.weight_decay}_np{args.num_parts}'
    elif args.method == "nodeformer_encoderfc":
        ckpt_name = f'bz{args.batch_size}_M{args.M}_K{args.K}_nl{args.num_layers}_hc{args.hidden_channels}_' \
                    f'nh{args.num_heads}_lmd{args.lamda}_ro{args.rb_order}_ug{args.use_gumbel}_ub{args.use_bn}_' \
                    f'ur{args.use_residual}_ua{args.use_act}_uj{args.use_jk}_rt{args.rb_trans}_lr{args.lr}_' \
                    f'wd{args.weight_decay}_np{args.num_parts}_wue{args.warmup_epoch}'
    ckpt_name += '.pkl'
    ckpt_path = os.path.join(save_dir, ckpt_name)
    state_dict = torch.load(ckpt_path, map_location=device)
    model = model.load_state_dict(state_dict)
    return model
