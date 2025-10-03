#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from log import log, WARNING, ERROR, INFO

class ModelOpts:
    def __init__(self,
                 in_channels=1,
                 out_channels1=64,
                 out_channels2=32,
                 out_channels3=32,
                 prior_channels1=16,
                 prior_channels2=32,
                 prior_channels3=32,
                 fc_layers=3,
                 fc_units=None,
                 out_dim=1,
                 kernel_size=None,
                 prior_kernel_size=None,
                 dropout_prob=0,
                 prior_features=None):
        if fc_units is None:
            fc_units = [128, 64, 32]
        if kernel_size is None:
            kernel_size = [11, 11, 11]
        if prior_kernel_size is None:
            prior_kernel_size = [3, 3, 3]
        
        self.in_channels = in_channels
        self.out_channels1 = out_channels1
        self.out_channels2 = out_channels2
        self.out_channels3 = out_channels3
        self.prior_channels1 = prior_channels1
        self.prior_channels2 = prior_channels2
        self.prior_channels3 = prior_channels3
        self.fc_layers = fc_layers
        self.fc_units = fc_units
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.prior_kernel_size = prior_kernel_size
        self.dropout_prob = dropout_prob
        self.prior_features = prior_features

class PKDP(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, out_channels3, 
                 prior_channels1, prior_channels2, prior_channels3,
                 fc_layers, fc_units, out_dim, input_length, 
                 kernel_sizes=None, prior_kernel_sizes=None, 
                 dropout_prob=0.3, prior_features=None, feature_names=None):
        super(PKDP, self).__init__()
        
        self.seq_length = input_length
        self.prior_features = prior_features
        self.feature_names = feature_names if feature_names is not None else [str(i) for i in range(input_length)]
        
        default_kernel_sizes = [11, 11, 11]
        default_prior_kernel_sizes = [3, 3, 3]
        
        if kernel_sizes is None or len(kernel_sizes) == 0:
            kernel_sizes = default_kernel_sizes
        elif len(kernel_sizes) == 1:
            kernel_sizes = [kernel_sizes[0]] * 3
        elif len(kernel_sizes) == 2:
            kernel_sizes = [kernel_sizes[0], kernel_sizes[1], kernel_sizes[1]]
        
        if prior_kernel_sizes is None or len(prior_kernel_sizes) == 0:
            prior_kernel_sizes = default_prior_kernel_sizes
        elif len(prior_kernel_sizes) == 1:
            prior_kernel_sizes = [prior_kernel_sizes[0]] * 3
        elif len(prior_kernel_sizes) == 2:
            prior_kernel_sizes = [prior_kernel_sizes[0], prior_kernel_sizes[1], prior_kernel_sizes[1]]
        
        for i in range(len(kernel_sizes)):
            if kernel_sizes[i] % 2 == 0:
                kernel_sizes[i] += 1
                log(WARNING, f"Adjusted main kernel size to odd number: {kernel_sizes[i]}")
        
        for i in range(len(prior_kernel_sizes)):
            if prior_kernel_sizes[i] % 2 == 0:
                prior_kernel_sizes[i] += 1
                log(WARNING, f"Adjusted prior kernel size to odd number: {prior_kernel_sizes[i]}")
        
        self.prior_indices = self._get_prior_indices()
        
        self.prior_path = nn.Sequential(
            nn.Conv1d(in_channels, prior_channels1, kernel_size=prior_kernel_sizes[0], padding=prior_kernel_sizes[0]//2),
            nn.BatchNorm1d(prior_channels1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(prior_channels1, prior_channels2, kernel_size=prior_kernel_sizes[1], padding=prior_kernel_sizes[1]//2),
            nn.BatchNorm1d(prior_channels2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(prior_channels2, prior_channels3, kernel_size=prior_kernel_sizes[2], padding=prior_kernel_sizes[2]//2),
            nn.BatchNorm1d(prior_channels3),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.general_path = nn.Sequential(
            nn.Conv1d(in_channels, out_channels1, kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2),
            nn.BatchNorm1d(out_channels1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(out_channels1, out_channels2, kernel_size=kernel_sizes[1], padding=kernel_sizes[1]//2),
            nn.BatchNorm1d(out_channels2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(out_channels2, out_channels3, kernel_size=kernel_sizes[2], padding=kernel_sizes[2]//2),
            nn.BatchNorm1d(out_channels3),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.fusion = nn.Conv1d(out_channels3 + prior_channels3, out_channels3, kernel_size=1)
        
        self._to_linear = None
        self._compute_conv_output_size((1, in_channels, input_length))
        
        fc_layers_list = []
        input_dim = self._to_linear
        for i in range(fc_layers):
            units = fc_units[i] if i < len(fc_units) else fc_units[-1]
            fc_layers_list.append(nn.Linear(input_dim, units))
            fc_layers_list.append(nn.ReLU())
            fc_layers_list.append(nn.Dropout(dropout_prob))
            input_dim = units
        fc_layers_list.append(nn.Linear(input_dim, out_dim))
        self.fc_layers = nn.Sequential(*fc_layers_list)
        
    def _get_prior_indices(self):
        if self.prior_features is None:
            return []
            
        indices = []
        missing_features = []
        
        if isinstance(self.prior_features, str):
            prior_ids = self.prior_features.split()
        elif isinstance(self.prior_features, (list, tuple)):
            prior_ids = self.prior_features
        else:
            return []

        import re
        name_to_idx = {}
        for idx, name in enumerate(self.feature_names):
            name_to_idx[name] = idx
        
        count_matched = 0
        for pid in prior_ids:
            try:
                if pid.isdigit():
                    idx = int(pid)
                    if 0 <= idx < self.seq_length:
                        indices.append(idx)
                        count_matched += 1
                    else:
                        pass
                else:
                    idx = name_to_idx.get(pid, -1)
                    if idx >= 0:
                        indices.append(idx)
                        count_matched += 1
                    else:
                        clean_pid = re.sub(r'[^0-9a-zA-Z_]', '_', pid)
                        idx = name_to_idx.get(clean_pid, -1)
                        if idx >= 0:
                            indices.append(idx)
                            count_matched += 1
                        else:
                            missing_features.append(pid)
            except ValueError:
                pass
        
        if missing_features and count_matched == 0 and len(prior_ids) > 0:
            pass
            
        return indices

    def _compute_conv_output_size(self, shape):
        x = torch.rand(shape)
        
        # general_out = self.general_path(x)
        general_x = x.clone()
        if self.prior_indices:
            for idx in self.prior_indices:
                if idx < x.shape[2]:
                    general_x[:, :, idx] = 0
        general_out = self.general_path(general_x)

        if self.prior_indices:
            prior_x = torch.zeros_like(x)
            for idx in self.prior_indices:
                if idx < x.shape[2]:
                    prior_x[:, :, idx] = x[:, :, idx]
            prior_out = self.prior_path(prior_x)
        else:
            prior_out = self.prior_path(x)
        
        combined = torch.cat([general_out, prior_out], dim=1)
        fused = self.fusion(combined)
        
        self._to_linear = fused.view(fused.size(0), -1).shape[1]

    def forward(self, x):
        # general_features = self.general_path(x)
        
        general_x = x.clone()

        if self.prior_indices:
            for idx in self.prior_indices:
                if idx < x.shape[2]:
                    general_x[:, :, idx] = 0
        general_features = self.general_path(general_x)
        
        if self.prior_indices:
            prior_x = torch.zeros_like(x)
            for idx in self.prior_indices:
                if idx < x.shape[2]:
                    prior_x[:, :, idx] = x[:, :, idx]
            prior_features = self.prior_path(prior_x)
            
            combined = torch.cat([general_features, prior_features], dim=1)
            fused = self.fusion(combined)
        else:
            fused = general_features
        
        out = fused.view(fused.size(0), -1)
        out = self.fc_layers(out)
        return out

def create_model(
    opts: ModelOpts = None,
    input_length: int = None,
    in_channels=None,
    out_channels1=None,
    out_channels2=None,
    out_channels3=None,
    prior_channels1=None,
    prior_channels2=None,
    prior_channels3=None,
    fc_layers=None,
    fc_units=None,
    out_dim=None,
    kernel_size=None,
    prior_kernel_size=None,
    dropout_prob=None,
    prior_features=None,
    feature_names=None
):
    if opts is None:
        opts = ModelOpts()
        
    if in_channels is not None:
        opts.in_channels = in_channels
    if out_channels1 is not None:
        opts.out_channels1 = out_channels1
    if out_channels2 is not None:
        opts.out_channels2 = out_channels2
    if out_channels3 is not None:
        opts.out_channels3 = out_channels3
    if prior_channels1 is not None:
        opts.prior_channels1 = prior_channels1
    if prior_channels2 is not None:
        opts.prior_channels2 = prior_channels2
    if prior_channels3 is not None:
        opts.prior_channels3 = prior_channels3
    if fc_layers is not None:
        opts.fc_layers = fc_layers
    if fc_units is not None:
        opts.fc_units = fc_units
    if out_dim is not None:
        opts.out_dim = out_dim
    if dropout_prob is not None:
        opts.dropout_prob = dropout_prob
    if prior_features is not None:
        opts.prior_features = prior_features

    if kernel_size is not None:
        if isinstance(kernel_size, (int, float)):
            opts.kernel_size = [kernel_size] * 3
        else:
            opts.kernel_size = kernel_size
            
    if prior_kernel_size is not None:
        if isinstance(prior_kernel_size, (int, float)):
            opts.prior_kernel_size = [prior_kernel_size] * 3
        else:
            opts.prior_kernel_size = prior_kernel_size
    
    kernel_sizes = opts.kernel_size
    prior_kernel_sizes = opts.prior_kernel_size
    
    if hasattr(opts, 'conv_kernel_size') and opts.conv_kernel_size:
        for i in range(min(len(opts.conv_kernel_size), 3)):
            kernel_sizes[i] = opts.conv_kernel_size[i]
            
    if hasattr(opts, 'prior_kernel_size') and opts.prior_kernel_size:
        for i in range(min(len(opts.prior_kernel_size), 3)):
            prior_kernel_sizes[i] = opts.prior_kernel_size[i]
        
    if input_length is None:
        raise ValueError("input_length must be specified")
    
    return PKDP(
        in_channels=opts.in_channels,
        out_channels1=opts.out_channels1,
        out_channels2=opts.out_channels2,
        out_channels3=opts.out_channels3,
        prior_channels1=opts.prior_channels1,
        prior_channels2=opts.prior_channels2,
        prior_channels3=opts.prior_channels3,
        fc_layers=opts.fc_layers,
        fc_units=opts.fc_units,
        out_dim=opts.out_dim,
        input_length=input_length,
        kernel_sizes=kernel_sizes,
        prior_kernel_sizes=prior_kernel_sizes,
        dropout_prob=opts.dropout_prob,
        prior_features=opts.prior_features,
        feature_names=feature_names
    )

def create_optimizer(model, learning_rate, optimizer_type='Adam'):
    if optimizer_type == 'SGD':
        return optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'Adam':
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'AdamW':
        return optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        log(ERROR, f"Unsupported optimizer type: {optimizer_type}")
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

def create_loss_function():
    return nn.MSELoss()