import argparse

import torch
from bert_score import BERTScorer

from protocol import serve


parser = argparse.ArgumentParser()
parser.add_argument("--path", required=True)
parser.add_argument("--num_layers", type=int, default=17)
args = parser.parse_args()

scorer = BERTScorer(
    model_type=args.path,
    num_layers=args.num_layers,
    batch_size=1,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    rescale_with_baseline=False,
)


def score(payload):
    references = payload["ref"]
    if not isinstance(references, list):
        references = [references]
    precision, recall, f1 = scorer.score(
        [str(payload["pred"])],
        [[str(reference) for reference in references]],
    )
    return {
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1[0]),
    }


serve(score)
