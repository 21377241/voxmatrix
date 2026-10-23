# Step-Audio-2 inference subset

This directory contains the inference files used by the VoxMatrix Step-Audio-2
adapter. `main.py` is VoxMatrix code; everything else is copied unmodified from
the upstream projects below.

| Files | Upstream | License |
|---|---|---|
| `stepaudio2.py`, `utils.py`, `token2wav.py`, `cosyvoice2/`, `assets/*.wav` | <https://github.com/stepfun-ai/Step-Audio2> | Apache-2.0 (`LICENSE`) |
| `flashcosyvoice/` (HiFT vocoder and mel utilities only) | <https://github.com/xingchensong/FlashCosyVoice> | Apache-2.0 (`flashcosyvoice/LICENSE`) |
