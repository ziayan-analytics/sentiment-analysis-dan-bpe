# Sentiment Analysis using Deep Averaging Networks + BPE

This project implements sentiment classification using Deep Averaging Networks (DAN) and subword tokenization via Byte Pair Encoding (BPE), comparing word-level and subword-level representations.

- Bag-of-Words (BOW) baselines  
- Deep Averaging Network (DAN) with pretrained GloVe embeddings  
- Subword DAN using Byte Pair Encoding (BPE) with randomly initialized embeddings  
- Experiments with different BPE vocabulary sizes  
- Skip-gram theoretical analysis

---

## Environment

- Python 3
- PyTorch
- sklearn
- matplotlib

All experiments were run inside the provided course environment.

---

## Directory Structure

Trains 2-layer and 3-layer BOW models and reports train/dev accuracy.

## Part 1 — Deep Averaging Network (DAN)

Run:

``` bash

python main.py --model DAN
```

Uses pretrained GloVe embeddings (trainable).

Typical dev accuracy: ~0.79–0.80.
---

### Part 2 — Subword DAN (BPE)

Run with default vocab size (1000):

```bash
python main.py --model SUBWORDDAN
```

Try different BPE sizes:
```bash
python main.py --model SUBWORDDAN --bpe_vocab_size 500
python main.py --model SUBWORDDAN --bpe_vocab_size 1000
python main.py --model SUBWORDDAN --bpe_vocab_size 3000

```
Pipeline:

1.Train BPE on training words

2.Build subword vocabulary

3.Initialize embeddings randomly

4.Train Subword DAN

Example dev accuracy:

500: ~0.65

1000: ~0.64

3000: ~0.65

Subword DAN performs worse than word-level DAN because embeddings are trained from scratch.
---
## Part 3 — Skip-Gram

Part 3 contains theoretical derivations:

-Training pairs

-Maximum likelihood distributions

-Constructed embeddings

See written report for details.
---
## Summary

-Word-level DAN with pretrained embeddings achieves best performance.

-Subword DAN works correctly but underperforms due to random embeddings.

-Increasing BPE vocab size gives limited improvement.
---
## How to Run
``` bash

python main.py --model BOW
python main.py --model DAN
python main.py --model SUBWORDDAN --bpe_vocab_size 1000
```
