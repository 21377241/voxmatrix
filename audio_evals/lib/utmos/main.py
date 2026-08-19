import argparse
import os
import select
import sys
import torchaudio

import torch

device = "cpu"


def load_model(path, source_path):
    if os.path.isfile(path):
        if not source_path:
            raise ValueError("source_path is required for a SpeechMOS checkpoint")
        sys.path.insert(0, source_path)
        from speechmos.utmos22.strong.model import UTMOS22Strong

        model = UTMOS22Strong()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model.eval().to(device), "speechmos"

    import lightning_module

    ssl_model = os.path.join(path, "wav2vec_small.pt")
    os.environ["SSL_MODEL_PATH"] = ssl_model
    model = lightning_module.BaselineLightningModule.load_from_checkpoint(
        os.path.join(path, "epoch=3-step=7459.ckpt"), map_location="cpu"
    )
    return model.eval().to(device), "space"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--source_path", type=str, default="", help="Path to SpeechMOS source"
    )
    config = parser.parse_args()

    model, backend = load_model(config.path, config.source_path)
    print("Model loaded from checkpoint: {}".format(config.path))

    while True:
        prompt = input()
        anchor = prompt.find("->")
        if anchor == -1:
            print(
                "Error: Invalid conversation format, must contains  ->, but {}".format(
                    prompt
                ),
                flush=True,
            )
            continue
        prefix = prompt[:anchor].strip() + "->"
        try:
            wav, sr = torchaudio.load(prompt[anchor + 2 :])
            wavs = wav.to(device)
            if len(wavs.shape) == 1:
                wavs = wavs.unsqueeze(0).unsqueeze(0)
            elif len(wavs.shape) == 2:
                wavs = wavs.mean(dim=0, keepdim=True)
                wavs = wavs.unsqueeze(0)
            elif len(wavs.shape) != 3:
                raise ValueError("Dimension of input tensor needs to be <= 3.")

            if sr != 16000:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=16000,
                    resampling_method="sinc_interpolation",
                    lowpass_filter_width=6,
                    dtype=torch.float32,
                ).to(device)
                wavs = resampler(wavs)

            batch = {
                "wav": wavs,
                "domains": torch.zeros(wavs.size(0), dtype=torch.int).to(device),
                "judge_id": torch.ones(wavs.size(0), dtype=torch.int).to(device) * 288,
            }

            with torch.no_grad():
                if backend == "speechmos":
                    score = model(wavs.squeeze(1), sr).mean().cpu().item()
                else:
                    output = model(batch)
                    score = output.mean(dim=1).squeeze(1).cpu().item() * 2 + 3

            retry = 3
            while retry:
                print(
                    "{}{}".format(
                        prefix,
                        score,
                    ),
                    flush=True,
                )
                rlist, _, _ = select.select([sys.stdin], [], [], 1)
                if rlist:
                    finish = sys.stdin.readline().strip()
                    if finish == "{}close".format(prefix):
                        break
                print("not found close signal, will emit again", flush=True)
                retry -= 1
        except Exception as e:
            import traceback

            traceback.print_exc()
            print("Error:{}".format(e))
