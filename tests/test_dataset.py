from audio_evals.dataset import huggingface as hf
from audio_evals.registry import registry


def test_huggingface_dataset_registry_load_is_bounded(monkeypatch):
    captured = {}

    def fake_load(name, subset, split, local_path, col_aliases, limit, **kwargs):
        captured.update(
            {
                "name": name,
                "subset": subset,
                "split": split,
                "local_path": local_path,
                "col_aliases": col_aliases,
                "limit": limit,
                "offset": kwargs["offset"],
            }
        )
        return [{"Text": "hello"}], {"row_count": 1}

    monkeypatch.setattr(hf, "load_audio_hf_dataset", fake_load)
    dataset = registry.get_dataset("KeSpeech")
    assert dataset.load(limit=1) == [{"Text": "hello"}]
    assert captured == {
        "name": "TwinkStart/kespeech",
        "subset": None,
        "split": "test",
        "local_path": "",
        "col_aliases": {},
        "limit": 1,
        "offset": 0,
    }
