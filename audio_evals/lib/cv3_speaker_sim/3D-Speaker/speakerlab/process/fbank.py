# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker).
# Licensed under the Apache License, Version 2.0.

import torchaudio.compliance.kaldi as Kaldi


class FBank:
    """Minimal inference-only filter-bank extractor used by VoxMatrix."""

    def __init__(self, n_mels, sample_rate, mean_nor: bool = False):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mean_nor = mean_nor

    def __call__(self, wav, dither=0):
        if self.sample_rate != 16000:
            raise ValueError("the bundled ERes2Net frontend requires 16 kHz audio")
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
        if len(wav.shape) == 3:
            wav = wav.squeeze(0)
        if wav.shape[0] > 1:
            wav = wav[0, :].unsqueeze(0)
        if len(wav.shape) != 2 or wav.shape[0] != 1:
            raise ValueError(f"expected waveform shape [1, T], got {tuple(wav.shape)}")
        feat = Kaldi.fbank(
            wav,
            num_mel_bins=self.n_mels,
            sample_frequency=self.sample_rate,
            dither=dither,
        )
        if self.mean_nor:
            feat = feat - feat.mean(0, keepdim=True)
        return feat
