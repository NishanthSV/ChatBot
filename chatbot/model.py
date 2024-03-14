import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_heads, hidden_size, num_layers, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.positional_encoding = PositionalEncoding(embedding_size, dropout)
        
        self.encoder_layers = nn.Sequential(*[
            EncoderLayer(embedding_size, num_heads, hidden_size, dropout) 
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(embedding_size, vocab_size)
        
    def forward(self, src, src_mask):
        src_embedding = self.embedding(src) * (self.embedding_size ** 0.5)
        src_embedding = self.positional_encoding(src_embedding)
        
        for layer in self.encoder_layers:
            src_embedding = layer(src_embedding, src_mask)
        
        output = self.fc_out(src_embedding)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, embedding_size, num_heads, hidden_size, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embedding_size, num_heads)
        self.feed_forward = FeedForward(embedding_size, hidden_size, dropout)
        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.layer_norm2 = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        src = self.layer_norm1(src + self.dropout(self.self_attention(src, src, src, src_mask)))
        src = self.layer_norm2(src + self.dropout(self.feed_forward(src)))
        return src

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert embedding_size % num_heads == 0
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_size = embedding_size // num_heads
        
        self.query = nn.Linear(embedding_size, embedding_size)
        self.key = nn.Linear(embedding_size, embedding_size)
        self.value = nn.Linear(embedding_size, embedding_size)
        self.fc_out = nn.Linear(embedding_size, embedding_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        query = self.query(query).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        key = self.key(key).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        value = self.value(value).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_size ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_size)
        output = self.fc_out(output)
        return output

class FeedForward(nn.Module):
    def __init__(self, embedding_size, hidden_size, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2) * -(math.log(10000.0) / embedding_size))
        self.positional_encoding = torch.zeros(max_len, embedding_size)
        self.positional_encoding[:, 0::2] = torch.sin(position * div_term)
        self.positional_encoding[:, 1::2] = torch.cos(position * div_term)
        self.positional_encoding = self.positional_encoding.unsqueeze(0)
        
    def forward(self, x):
        x = x + self.positional_encoding[:, :x.size(1)]
        return self.dropout(x)