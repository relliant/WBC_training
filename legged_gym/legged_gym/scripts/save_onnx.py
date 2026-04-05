#!/usr/bin/env python3
"""
ONNX conversion script for g1_stu_future (student policy with future motion support)
Usage: python save_onnx_stu_future.py --ckpt_path <absolute_path_to_checkpoint>
"""

import os, sys
sys.path.append("../../../rsl_rl")
import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic_future import ActorFuture, get_activation
import argparse
from termcolor import cprint
import numpy as np
import re

class HardwareStudentFutureNN(nn.Module):
    """Hardware deployment wrapper for student policy with future motion support."""
    
    def __init__(self,  
                 num_observations,
                 num_motion_observations,
                 num_priop_observations,
                 num_motion_steps,
                 num_future_observations,
                 num_future_steps,
                 motion_latent_dim,
                 future_latent_dim,
                 num_actions,
                 actor_hidden_dims,
                 activation,
                 history_latent_dim,
                 num_history_steps,
                 layer_norm=False,
                 tanh_encoder_output=False,
                 **kwargs):
        super().__init__()

        self.num_observations = num_observations
        self.num_actions = num_actions
        self.num_motion_observations = num_motion_observations
        self.num_priop_observations = num_priop_observations
        
        activation = get_activation(activation)
        
        self.normalizer = None
        
        self.actor = ActorFuture(
            num_observations=num_observations,
            num_motion_observations=num_motion_observations,
            num_priop_observations=num_priop_observations,
            num_motion_steps=num_motion_steps,
            num_future_observations=num_future_observations,
            num_future_steps=num_future_steps,
            motion_latent_dim=motion_latent_dim,
            future_latent_dim=future_latent_dim,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            activation=activation,
            history_latent_dim=history_latent_dim,
            num_history_steps=num_history_steps,
            layer_norm=layer_norm,
            tanh_encoder_output=tanh_encoder_output,
            **kwargs
        )

    def load_normalizer(self, normalizer):
        self.normalizer = normalizer

    def forward(self, obs):
        assert obs.shape[1] == self.num_observations, f"Expected {self.num_observations} but got {obs.shape[1]}"
        obs = self.normalizer.normalize(obs)
        return self.actor(obs)


def infer_model_config_from_checkpoint(ac_state_dict):
    """Infer student-future model dimensions directly from checkpoint tensors."""
    model_state_dict = ac_state_dict['model_state_dict']

    # Core dimensions from first-layer weights of each encoder/backbone.
    num_motion_observations = model_state_dict['actor.motion_encoder.encoder.0.weight'].shape[1]
    single_history_obs = model_state_dict['actor.history_encoder.encoder.0.weight'].shape[1]
    num_priop_observations = single_history_obs - num_motion_observations
    # num_actions is inferred from the last linear layer in actor_backbone.
    num_actions = None

    # FutureMotionEncoder flattens (steps * (single_future_obs - 1)) into Linear input.
    future_flat_obs = model_state_dict['actor.future_encoder.encoder.0.weight'].shape[1]
    num_future_steps = 1
    num_future_observations = future_flat_obs + 1

    # History length from total observation dimension saved by normalizer.
    if ac_state_dict.get('normalizer') is None:
        raise ValueError("Checkpoint does not include normalizer; cannot infer num_observations.")
    normalizer_state = ac_state_dict['normalizer'].state_dict()
    num_observations = int(normalizer_state['_mean'].shape[0])

    current_obs = num_motion_observations + num_priop_observations
    history_numerator = num_observations - current_obs - num_future_observations
    if history_numerator < 0 or history_numerator % current_obs != 0:
        raise ValueError(
            f"Failed to infer history_len from checkpoint: num_obs={num_observations}, "
            f"current_obs={current_obs}, future_obs={num_future_observations}"
        )
    history_len = history_numerator // current_obs

    # Hidden dims from 2D linear layers in actor_backbone (ignore LayerNorm 1D weights).
    backbone_linear = []
    for key, value in model_state_dict.items():
        match = re.match(r'actor\.actor_backbone\.(\d+)\.weight$', key)
        if match and value.dim() == 2:
            backbone_linear.append((int(match.group(1)), value.shape))
    backbone_linear.sort(key=lambda x: x[0])
    if len(backbone_linear) < 2:
        raise ValueError("Unexpected actor_backbone structure in checkpoint.")
    actor_hidden_dims = [int(shape[0]) for _, shape in backbone_linear[:-1]]
    num_actions = int(backbone_linear[-1][1][0])

    return {
        'num_observations': num_observations,
        'num_motion_observations': int(num_motion_observations),
        'num_priop_observations': int(num_priop_observations),
        'num_motion_steps': 1,
        'num_future_observations': int(num_future_observations),
        'num_future_steps': int(num_future_steps),
        'num_actions': int(num_actions),
        'history_len': int(history_len),
        'actor_hidden_dims': actor_hidden_dims,
        'motion_latent_dim': int(model_state_dict['actor.motion_encoder.linear_output.weight'].shape[0]),
        'history_latent_dim': int(model_state_dict['actor.history_encoder.linear_output.weight'].shape[0]),
        'future_latent_dim': int(model_state_dict['actor.future_encoder.encoder.6.weight'].shape[0]),
    }

def convert_to_onnx(args):
    """Convert g1_stu_future student policy to ONNX."""
    
    ckpt_path = args.ckpt_path
    
    # Check if checkpoint file exists
    if not os.path.exists(ckpt_path):
        cprint(f"Error: Checkpoint file not found: {ckpt_path}", "red")
        return
    
    # Load trained model first so we can infer architecture from checkpoint.
    cprint(f"Loading model from: {ckpt_path}", "green")
    device = torch.device('cuda')
    ac_state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = infer_model_config_from_checkpoint(ac_state_dict)
    activation = 'silu'

    n_obs_single = cfg['num_motion_observations'] + cfg['num_priop_observations']
    print("Student Future Policy Configuration (inferred from checkpoint):")
    print(f"  Actions: {cfg['num_actions']}")
    print(f"  History length: {cfg['history_len']}")
    print(f"  Motion observations: {cfg['num_motion_observations']}")
    print(f"  Proprioceptive observations: {cfg['num_priop_observations']}")
    print(f"  Future observations: {cfg['num_future_observations']}")
    print(f"  Single obs size: {n_obs_single}")
    print(f"  Total observations: {cfg['num_observations']}")
    print(f"  Motion latent dim: {cfg['motion_latent_dim']}")
    print(f"  Future latent dim: {cfg['future_latent_dim']}")
    print(f"  History latent dim: {cfg['history_latent_dim']}")
    print(f"  Actor hidden dims: {cfg['actor_hidden_dims']}")
    print("")

    policy = HardwareStudentFutureNN(
        num_observations=cfg['num_observations'],
        num_motion_observations=cfg['num_motion_observations'],
        num_priop_observations=cfg['num_priop_observations'],
        num_motion_steps=cfg['num_motion_steps'],
        num_future_observations=cfg['num_future_observations'],
        num_future_steps=cfg['num_future_steps'],
        motion_latent_dim=cfg['motion_latent_dim'],
        future_latent_dim=cfg['future_latent_dim'],
        num_actions=cfg['num_actions'],
        actor_hidden_dims=cfg['actor_hidden_dims'],
        activation=activation,
        history_latent_dim=cfg['history_latent_dim'],
        num_history_steps=cfg['history_len'],
        layer_norm=True,
        tanh_encoder_output=False,
        use_history_encoder=True,
        use_motion_encoder=True
        
    ).to(device)

    policy.load_state_dict(ac_state_dict['model_state_dict'], strict=False)
    policy.load_normalizer(ac_state_dict['normalizer'])
    
    policy = policy.to(device)
    
    # Export to ONNX with same name but .onnx extension
    policy.eval()
    with torch.no_grad(): 
        # Create dummy input with correct observation structure
        batch_size = 1  # Use batch size 1 for simplicity
        obs_input = torch.ones(batch_size, cfg['num_observations'], device=device)
        cprint(f"Input observation shape: {obs_input.shape}", "cyan")
        
        # Generate ONNX path with same name but .onnx extension
        onnx_path = ckpt_path.replace('.pt', '.onnx')
        
        # Export to ONNX
        torch.onnx.export(
            policy,
            obs_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        cprint(f"ONNX model saved to: {onnx_path}", "green")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert g1_stu_future student policy to ONNX')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Absolute path to checkpoint file')
    args = parser.parse_args()
    convert_to_onnx(args)