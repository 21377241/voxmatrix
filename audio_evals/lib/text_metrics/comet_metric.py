import argparse

import torch
from comet.models import RegressionMetric

from protocol import serve


parser = argparse.ArgumentParser()
parser.add_argument("--path", required=True)
parser.add_argument("--encoder_path", required=True)
args = parser.parse_args()

model = RegressionMetric.load_from_checkpoint(
    checkpoint_path=args.path,
    map_location="cpu",
    load_pretrained_weights=False,
    pretrained_model=args.encoder_path,
    strict=False,
)


def score(payload):
    data = [
        {
            "src": str(payload["src"]),
            "mt": str(payload["mt"]),
            "ref": str(payload["ref"]),
        }
    ]
    output = model.predict(
        data,
        batch_size=1,
        gpus=1 if torch.cuda.is_available() else 0,
        progress_bar=False,
    )
    return {"score": float(output.system_score)}


serve(score)
