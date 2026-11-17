# my-gpt

Personal learning project: build a small character-level language model in PyTorch, step by step, while following Andrej Karpathy’s GPT/nanoGPT material and writing my own implementation.

The project started with a bigram model and now uses token and position embeddings, causal multi-head self-attention, and a feedforward network. Training and sampling currently live in a notebook; this is an early GPT-style model, with more of the Transformer architecture still to implement.

## What is implemented

- **Dataset:** Tiny Shakespeare is included at `datasets/tinyshakespear.txt` (1,115,394 characters).
- **Character tokenization:** the notebook builds a vocabulary of 65 unique characters and maps them to consecutive integer IDs, with encoding and decoding helpers.
- **Data preparation:** a sequential 90% training / 10% held-out split, with random batches of 64-character contexts and targets shifted by one character. The held-out split is named `test` in the code and is used during training evaluation.
- **Model:** learned token and position embeddings, four causal attention heads, a two-layer feedforward network with ReLU activations, and a final projection to vocabulary logits.
- **Training:** next-character prediction with cross-entropy loss, backpropagation, and the AdamW optimizer.
- **Evaluation:** mean training and held-out losses over multiple random batches.
- **Sampling:** autoregressive generation using the most recent context and multinomial sampling from next-character probabilities.
- **Device selection:** the notebook selects CUDA when available and otherwise uses the CPU.

## Current model

`SimpleModel` in [model.py](model.py) follows this sequence:

```text
Character IDs
  → token embeddings + position embeddings
  → causal multi-head self-attention
  → feedforward network
  → vocabulary logits
```

The notebook uses these settings:

| Setting | Value |
| --- | --- |
| Vocabulary size | 65 |
| Context length | 64 characters |
| Embedding size | 128 |
| Attention heads | 4 (32 dimensions each) |
| Feedforward network | Two 128 → 128 linear layers, each followed by ReLU |
| Batch size | 64 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Configured training steps | 5,000 |
| Evaluation | Every 200 steps, averaging 200 batches per split |

This is one attention/feedforward stage. Residual connections, layer normalization, dropout, an attention output projection, and stacked Transformer blocks are not implemented yet.

## Run the notebook

Use Python 3.12 or newer and `uv`. From the repository root:

```bash
uv sync
uv run --with jupyterlab --with ipykernel jupyter lab
```

Open [protoyping.ipynb](protoyping.ipynb) (the filename currently has this spelling) and run the cells in order through training. The dataset is already included, so no download is needed. PyTorch and NumPy are declared in `pyproject.toml`; the command above supplies Jupyter for the session.

For sampling, skip the older cell that starts with `length = 1000` and creates `curr_tok`: it passes a one-dimensional tensor to a model that expects a batch and sequence dimension. Use the following cell, which calls `model.generate(context_x, length)` with batched contexts from `get_batch('test', ...)`.

`main.py` is still a greeting-only placeholder; it does not train the model. There is no training command-line interface yet.

## Recorded progress

The notebook contains saved training output showing the following cross-entropy losses:

| Step | Training loss | Held-out loss (`test`) |
| --- | --- | --- |
| 0 | 4.1710 | 4.1712 |
| 1,000 | 1.9615 | 2.0641 |
| 2,000 | 1.7693 | 1.9426 |
| 3,000 | 1.6947 | 1.8694 |

These are saved notebook results, not a newly reproduced benchmark. Although the loop is configured for 5,000 steps, the saved log stops at step 3,000 and does not include the final loss report. Saved samples contain word-like fragments and dialogue formatting, but are still noisy and often incoherent.

## Repository layout

```text
datasets/tinyshakespear.txt  Included training text
protoyping.ipynb            Data preparation, training, evaluation, and sampling
model.py                   Attention, feedforward network, and SimpleModel
utils/tokenizer.py         Separate ord/chr helpers (not the notebook tokenizer)
utils/estimate_loss.py     Incomplete standalone evaluation helper
main.py                    Placeholder entry point
pyproject.toml / uv.lock   Python dependencies and lockfile
```

## Current limitations and next steps

- **Generation retains only a rolling window.** `generate` truncates `idx` itself on every iteration, so it returns at most `context_size + 1` characters after generating, rather than the full prompt and continuation. Preserve the full sequence separately from the model’s input window.
- **Some utilities are unfinished.** `utils/estimate_loss.py` references `get_batch` and `F` without defining or importing them. The working evaluation function is in the notebook. The `ord`/`chr` helpers in `utils/tokenizer.py` do not use the notebook’s compact vocabulary IDs.
- **Training state is not saved.** Add checkpoint saving/loading for model weights, optimizer state, configuration, and vocabulary.
- **Complete the Transformer block.** Add residual connections, layer normalization, and attention output projection, then stack multiple blocks.
- **Make experiments easier to rerun.** Move the notebook workflow into a training script with configuration and reproducible seeds, and remove the obsolete sampling cell.

Once the small character-level pipeline is more complete, the longer-term plan is to experiment with a subword tokenizer and larger datasets/models.
