from typing import Any

import torch 
import torch.nn as nn



class AttentionHead(nn.Module):

    def __init__(self, n_embed, head_size, context_size):
        super().__init__()
        self.n_embed = n_embed
        self.head_size = torch.tensor(head_size)
        self.context_size = context_size
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))
 

    def forward(self, x):
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)
        w = query @ key.mT 
        w = w / torch.sqrt(self.head_size)
        w = w.masked_fill(self.tril[:w.size(1), :w.size(1)] == 0, float('-inf'))
        w = torch.softmax(w, dim=-1)
        attention = w @ value
        return attention


class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, n_embed, context_size):
        super().__init__()
        self.n_heads = n_heads
        self.n_embed = n_embed
        self.context_size = context_size
        self.head_size = n_embed // n_heads
        self.heads = nn.ModuleList([AttentionHead(n_embed, self.head_size, self.context_size) for _ in range(n_heads)])


    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return out
    

class FeedForward(nn.Module):

    def __init__(self, n_layers, embed_d, activation=nn.ReLU):
        super().__init__()
        self.n_layers = n_layers
        self.activation = activation
        self.embed_d = embed_d
        self.net = nn.Sequential()

        for _ in range(self.n_layers):
            self.net.append(nn.Linear(self.embed_d, self.embed_d))
            self.net.append(self.activation())

    def forward(self, x):
        return self.net(x)


class SimpleModel(nn.Module):

    def __init__(self, vocab_size, n_embd, context_size, n_heads):
        super().__init__()
        self.context_size = context_size
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(context_size, n_embd)
        self.multi_head = MultiHeadAttention(n_heads=n_heads, n_embed=n_embd, context_size=context_size)
        self.ffwd = FeedForward(n_layers=2, embed_d=n_embd)
        self.layer = nn.Linear(n_embd, vocab_size)  
        
    def forward(self, idx):
        # idx of shape Batch, Tokens, 
        tok_emb = self.tok_emb(idx)                                # shape B, T, n_embd each token in each batch get it's vector embedding
        pos_emb = self.pos_emb(torch.arange(idx.size(1), 
                                            device=idx.device))    # all position embeddings from 0 to T-1, shape T, n_embd

        x = pos_emb + tok_emb
        x = self.multi_head(x)
        x = self.ffwd(x)
        logits = self.layer(x)
        return logits 
    
    def generate(self, idx, max_new_tokens):

        for i in range(max_new_tokens):
            idx = idx[:,-self.context_size:]
            logits = self(idx) 
            last_logits = logits[:, -1, :] # The logits of the next token for the last token of each batch
            probs = torch.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, next_token), dim=1) 

        return idx
    
