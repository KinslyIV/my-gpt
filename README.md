# my-gpt

Personal project: implement a small GPT training stack inspired by Andrej Karpathy (nanoGPT), step-by-step.

## Plan (Hybrid learning)
- **nanoGPT mainline** for the full working training pipeline.
- **makemore refreshers** when a concept is unclear (tokenization, attention, block anatomy, training loop details).

## Staged approach
1) **Stage A (server, CPU):**
   - Start with **TinyShakespeare** and a **small model**.
   - Prove end-to-end: data → model → train → loss down → sample generation → checkpoints.
2) **Stage B:** swap tokenizer (e.g. tiktoken) while keeping dataset small to isolate changes.
3) **Stage C:** scale up dataset + model.

## Compute / topology
- Training GPU is on Boby’s **laptop**, not on this server.
- Start development/training on the **server** for simplicity (small runs).
- Later, use **Tailscale** for a private network between server and laptop (avoid opening public ports).

## Safety / ops rules (global)
- **Mode A:** always show exact shell commands before running them.
- Always ask for explicit approval before:
  - modifying files outside `/home/immata/.openclaw/workspace`
  - interacting with processes
  - interacting with Docker containers
  - executing risky commands
- These rules also apply to spawned sub-agents.

## Working style (how we build this)
- We are **not vibe-coding**.
- Boby follows Andrej Karpathy’s nanoGPT video and writes his own implementation.
- Pixal32’s role: **explain concepts**, **help debug**, **sanity-check choices**, and **suggest next steps**—without taking over the build.

## Next steps
- Implement: CharTokenizer, dataset loader, minimal GPT model, training loop.
- Get a first run training and generating samples.
