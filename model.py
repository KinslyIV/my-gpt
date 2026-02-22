from typing import Any

import torch 
import torch.nn as nn



class AttentionHead(nn.Module):

    def __init__(self, n_embed, head_size):
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(n_embed, n_embed)))


    def forward(self, x):
        key = self.key(x)
        query = self.query(x)
        w = query @ key.mT
        return w



class SimpleModel(nn.Module):

    def __init__(self, vocab_size, n_embd, context_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(context_size, n_embd)
        self.layer = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx):
        # idx of shape Batch, Tokens, 
        tok_emb = self.tok_emb(idx)                                # shape B, T, n_embd each token in each batch get it's vector embedding
        pos_emb = self.pos_emb(torch.arange(idx.size(1), 
                                            device=idx.device))    # all position embeddings from 0 to T-1, shape T, n_embd

        x = pos_emb + tok_emb

        logits = self.layer(x)
        return logits 
    
    def generate(self, idx, max_new_tokens):

        for i in range(max_new_tokens):
            logits = self(idx)
            last_logits = logits[:, -1, :] # The logits of the next token for the last token of each batch
            probs = torch.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, next_token), dim=1) 

        return idx
    
