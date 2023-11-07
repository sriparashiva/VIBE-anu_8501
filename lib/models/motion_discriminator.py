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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from lib.models.attention import SelfAttention
import math

class MotionDiscriminator(nn.Module):

    def __init__(self,
                 rnn_size,
                 input_size,
                 num_layers,
                 output_size=2,
                 feature_pool="concat",
                 use_spectral_norm=False,
                 attention_size=1024,
                 attention_layers=1,
                 attention_dropout=0.5):

        super(MotionDiscriminator, self).__init__()
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.feature_pool = feature_pool
        self.num_layers = num_layers
        self.attention_size = attention_size
        self.attention_layers = attention_layers
        self.attention_dropout = attention_dropout

        self.gru = nn.GRU(self.input_size, self.rnn_size, num_layers=num_layers)

        linear_size = self.rnn_size if not feature_pool == "concat" else self.rnn_size * 2

        if feature_pool == "attention" :
            self.attention = SelfAttention(attention_size=self.attention_size,
                                       layers=self.attention_layers,
                                       dropout=self.attention_dropout)
        if use_spectral_norm:
            self.fc = spectral_norm(nn.Linear(linear_size, output_size))
        else:
            self.fc = nn.Linear(linear_size, output_size)

    def forward(self, sequence):
        """
        sequence: of shape [batch_size, seq_len, input_size]
        """
        batchsize, seqlen, input_size = sequence.shape
        sequence = torch.transpose(sequence, 0, 1)

        outputs, state = self.gru(sequence)

        if self.feature_pool == "concat":
            outputs = F.relu(outputs)
            avg_pool = F.adaptive_avg_pool1d(outputs.permute(1, 2, 0), 1).view(batchsize, -1)
            max_pool = F.adaptive_max_pool1d(outputs.permute(1, 2, 0), 1).view(batchsize, -1)
            output = self.fc(torch.cat([avg_pool, max_pool], dim=1))
        elif self.feature_pool == "attention":
            outputs = outputs.permute(1, 0, 2)
            y, attentions = self.attention(outputs)
            output = self.fc(y)
        else:
            output = self.fc(outputs[-1])
            
        return output

    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Correcting the calculation for div_term to handle odd d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        # Handling the possibility of d_model being odd
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            # Ensuring that we don't go out of bounds on the last cos
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class MotionDiscriminator_Transformer(nn.Module):
    def __init__(self,
             input_size,
             output_size=1,
             dim_feedforward=1024,
             attention_size=1024,
             attention_layers=1,
             attention_dropout=0.5,
             nhead=3, 
             num_layers=2,
             seqlen=16,
             dropout=0.3):

        super(MotionDiscriminator_Transformer, self).__init__()
        self.input_size = input_size
        self.attention_size = attention_size
        self.attention_layers = attention_layers
        self.attention_dropout = attention_dropout
                
        self.pos_encoder = PositionalEncoding(input_size, dropout, max_len=seqlen) # Adding dropout to positional encoding as well
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.feed_forward = nn.Linear(input_size, attention_size)
        
        self.attention = SelfAttention(attention_size=self.attention_size,
                                       layers=self.attention_layers,
                                       dropout=self.attention_dropout)
        
        self.fc = nn.Linear(attention_size, output_size)
        
        self._init_weights()
        
    def _init_weights(self):
        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))
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
        outputs = self.transformer_encoder(x)
        
        outputs = F.gelu(self.feed_forward(outputs))
        
        outputs = outputs.permute(1, 0, 2)
        
        y, attentions = self.attention(outputs)
        
        output = self.fc(y)

        return output