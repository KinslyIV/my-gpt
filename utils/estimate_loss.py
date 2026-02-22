import torch

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    losses = {}
    for split in ['train', 'test']:
        x, y = get_batch(split)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses[split] = loss.item()
    return losses