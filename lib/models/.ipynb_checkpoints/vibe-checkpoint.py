# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F

from lib.core.config import VIBE_DATA_DIR
from lib.models.spin import Regressor, hmr

import math
    
    
# class PositionWiseFeedForward(nn.Module):
#     def __init__(self, d_model, d_ff):
#         super(PositionWiseFeedForward, self).__init__()
#         self.fc1 = nn.Linear(d_model, d_model)
#         self.fc2 = nn.Linear(d_model, d_ff)

#     def forward(self, x):
#         return self.fc2(F.gelu(self.fc1(x)))

        
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         """
#         Arguments:
#             x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
#         """
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)

# class TemporalEncoderTransformer(nn.Module):
#     def __init__(self, 
#                  d_model=2048, 
#                  nhead=8, 
#                  num_encoder_layers=1,
#                  seqlen=16,
#                  dropout=0.1):
#         super(TemporalEncoderTransformer, self).__init__()
        
#         self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seqlen) # Adding dropout to positional encoding as well
        
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model, 
#             nhead=nhead, 
#             dim_feedforward=d_model,
#             dropout=dropout,
#             activation='gelu'
#         )
        
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
#         self.feed_forward = PositionWiseFeedForward(d_model, d_model)
        
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
        
#         self.dropout = nn.Dropout(dropout)
        
#         self._init_weights()
        
#     def _init_weights(self):
#         for p in self.transformer_encoder.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p, gain=nn.init.calculate_gain('gelu'))
#                 # nn.init.kaiming_uniform_(p, mode='fan_in')
#             elif p.dim() == 1:
#                 nn.init.constant_(p, 0)  # Biases are initialized to zero.
#         for p in self.feed_forward.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p, gain=nn.init.calculate_gain('gelu'))
#                 # nn.init.kaiming_uniform_(p, mode='fan_in')
#             else:
#                 nn.init.constant_(p, 0)

#     def forward(self, x):
#         # x: NTF -> x: TNF (sequence length first for transformer)
#         x = x.permute(1, 0, 2)
        
#         # Pass through transformer
#         transformer_output = self.transformer_encoder(self.pos_encoder(x))
        
#         transformer_output = self.norm1(self.dropout(transformer_output))
        
#         ff_output = self.feed_forward(transformer_output)
        
#         y = self.norm2(self.dropout(ff_output))
        
#         y = y + x
                
#         # y: TNF -> y: NTF
#         y = y.permute(1, 0, 2)
#         return y

        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return x

class TemporalEncoderTransformer(nn.Module):
    def __init__(self, 
                 d_model=2048, 
                 nhead=8, 
                 num_encoder_layers=2,
                 seqlen=16,
                 dropout=0.1):
        super(TemporalEncoderTransformer, self).__init__()
        
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seqlen) # Adding dropout to positional encoding as well
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model,
            dropout=dropout,
            activation='gelu',
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)        
        
        
        self._init_weights()
        
    def _init_weights(self):
        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))
                # nn.init.kaiming_uniform_(p, mode='fan_in')
            elif p.dim() == 1:
                nn.init.constant_(p, 0)  # Biases are initialized to zero.
            else:
                nn.init.constant_(p, 0)

    def forward(self, x):
        # x: NTF -> x: TNF (sequence length first for transformer)
        x = x.permute(1, 0, 2)

        # Add positional encoding to the input
        x = self.pos_encoder(x)

        # Pass through transformer encoder
        y = self.transformer_encoder(x)

        # y: TNF -> y: NTF
        y = y.permute(1, 0, 2)
        return y
    

#     def forward(self, x):
#         # x: NTF -> x: TNF (sequence length first for transformer)
#         x = x.permute(1, 0, 2)
        
#         # Add positional encodings
#         x = self.pos_encoder(x)
        
#         # Pass through transformer
#         transformer_output = self.transformer_encoder(x)
        
#         x = self.norm1(x + self.dropout(transformer_output))
        
#         ff_output = self.feed_forward(x)
        
#         y = self.norm2(x + self.dropout(ff_output))
                
#         # y: TNF -> y: NTF
#         y = y.permute(1, 0, 2)
#         return y

class TemporalEncoder(nn.Module):
    def __init__(
            self,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True
    ):
        super(TemporalEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=2048,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=n_layers
        )

        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(hidden_size*2, 2048)
        elif add_linear:
            self.linear = nn.Linear(hidden_size, 2048)
        self.use_residual = use_residual

    def forward(self, x):
        n,t,f = x.shape
        x = x.permute(1,0,2) # NTF -> TNF
        y, _ = self.gru(x)
        if self.linear:
            y = F.relu(y)
            y = self.linear(y.view(-1, y.size(-1)))
            y = y.view(t,n,f)
        if self.use_residual and y.shape[-1] == 2048:
            y = y + x
        y = y.permute(1,0,2) # TNF -> NTF
        return y


class VIBE(nn.Module):
    def __init__(
            self,
            seqlen,
            batch_size=64,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True,
            pretrained=osp.join(VIBE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
            temporal_type='gru',
            nhead=8,
            tform_n_layers=6,
            tform_dropout=0.1,
    ):

        super(VIBE, self).__init__()

        self.seqlen = seqlen
        self.batch_size = batch_size
        self.temporal_type = temporal_type
        
        if temporal_type == 'gru':
            self.encoder = TemporalEncoder(
                n_layers=n_layers,
                hidden_size=hidden_size,
                bidirectional=bidirectional,
                add_linear=add_linear,
                use_residual=use_residual,
            )
            print('Using GRU encoder for Temporal Encoder')
        elif temporal_type == 'transformer':
            self.encoder_transformer = TemporalEncoderTransformer(
                d_model=2048,
                nhead=nhead, 
                num_encoder_layers=tform_n_layers,
                seqlen=seqlen,
                dropout=tform_dropout
            )
            print('Using Transformer encoder for Temporal Encoder')
        else:
            print(f'Invalid temporal model selected: {temporal_type}')

        # regressor can predict cam, pose and shape params in an iterative way
        self.regressor = Regressor()

        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')


    def forward(self, input, J_regressor=None):
        
        # input size NTF
        batch_size, seqlen = input.shape[:2]
        
        
        if self.temporal_type == 'gru':
            feature = self.encoder(input)
        elif self.temporal_type == 'transformer':
            feature = self.encoder_transformer(input)
        
        feature = feature.reshape(-1, feature.size(-1))

        smpl_output = self.regressor(feature, J_regressor=J_regressor)
        for s in smpl_output:
            s['theta'] = s['theta'].reshape(batch_size, seqlen, -1)
            s['verts'] = s['verts'].reshape(batch_size, seqlen, -1, 3)
            s['kp_2d'] = s['kp_2d'].reshape(batch_size, seqlen, -1, 2)
            s['kp_3d'] = s['kp_3d'].reshape(batch_size, seqlen, -1, 3)
            s['rotmat'] = s['rotmat'].reshape(batch_size, seqlen, -1, 3, 3)

        return smpl_output


class VIBE_Demo(nn.Module):
    def __init__(
            self,
            seqlen,
            batch_size=64,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True,
            pretrained=osp.join(VIBE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
            temporal_type='gru',
            nhead=8,
            tform_n_layers=6,
            tform_dropout=0.1,
    ):

        super(VIBE_Demo, self).__init__()

        self.seqlen = seqlen
        self.batch_size = batch_size
        self.temporal_type = temporal_type

        if temporal_type == 'gru':
            self.encoder = TemporalEncoder(
                n_layers=n_layers,
                hidden_size=hidden_size,
                bidirectional=bidirectional,
                add_linear=add_linear,
                use_residual=use_residual,
            )
            print('Using GRU encoder for Temporal Encoder')
        elif temporal_type == 'transformer':
            self.encoder_transformer = TemporalEncoderTransformer(
                d_model=2048,
                nhead=nhead, 
                num_encoder_layers=tform_n_layers,
                seqlen=seqlen,
                dropout=tform_dropout
            )
            print('Using Transformer encoder for Temporal Encoder')
        else:
            print(f'Invalid temporal model selected: {temporal_type}')

        self.hmr = hmr()
        checkpoint = torch.load(pretrained)
        self.hmr.load_state_dict(checkpoint['model'], strict=False)

        # regressor can predict cam, pose and shape params in an iterative way
        self.regressor = Regressor()

        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')


    def forward(self, input, J_regressor=None):
        # input size NTF
        batch_size, seqlen, nc, h, w = input.shape

        feature = self.hmr.feature_extractor(input.reshape(-1, nc, h, w))

        feature = feature.reshape(batch_size, seqlen, -1)
        if self.temporal_type == 'gru':
            feature = self.encoder(input)
        elif self.temporal_type == 'transformer':
            feature = self.encoder_transformer(input)
        feature = feature.reshape(-1, feature.size(-1))

        smpl_output = self.regressor(feature, J_regressor=J_regressor)

        for s in smpl_output:
            s['theta'] = s['theta'].reshape(batch_size, seqlen, -1)
            s['verts'] = s['verts'].reshape(batch_size, seqlen, -1, 3)
            s['kp_2d'] = s['kp_2d'].reshape(batch_size, seqlen, -1, 2)
            s['kp_3d'] = s['kp_3d'].reshape(batch_size, seqlen, -1, 3)
            s['rotmat'] = s['rotmat'].reshape(batch_size, seqlen, -1, 3, 3)

        return smpl_output
