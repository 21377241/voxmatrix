import json
import os.path
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List
import pandas as pd
from tqdm import tqdm

tqdm.pandas()


class Dataset(ABC):
    def __init__(self, default_task: str, ref_col: str, col_aliases=None):
        if col_aliases is None:
            col_aliases = {}
        self.col_aliases = col_aliases
        self.task_name = default_task
        self.ref_col = ref_col

    def reset_ref_col(self, ref_col: str):
        self.ref_col = ref_col

    @abstractmethod
    def load(self, limit=0) -> List[Dict[str, any]]:
        raise NotImplementedError()

    def load_slice(self, offset=0, limit=0) -> List[Dict[str, any]]:
        """Load a deterministic contiguous slice.

        Dataset implementations that can seek efficiently should override this
        method.  The fallback preserves the historical ``load(limit)`` API while
        giving the execution layer one place to express shard offsets.
        """
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if limit < 0:
            raise ValueError("limit must be non-negative")
        load_limit = offset + limit if limit else 0
        data = self.load(load_limit)
        return data[offset : offset + limit if limit else None]

    def resume_from(self, f_name: str):
        from audio_evals.dataset.resume import ResumeDataset

        return ResumeDataset(self, f_name)

    def load_inf_file(self, f_name: str):
        from audio_evals.dataset.resume import ResumeDataset

        return ResumeDataset(self, f_name, save_type=["prompt", "inference"])


class JsonlFile(Dataset):
    def __init__(self, f_name: str, default_task: str, ref_col: str, col_aliases=None):
        super().__init__(default_task, ref_col, col_aliases)
        self.f_name = f_name

    def add_col_alias(self, df):
        for k, v in self.col_aliases.items():
            if v in df.columns:
                raise ValueError(f"Column alias {v} already exists in the dataframe")
            df[v] = df[k]
        return df

    def load(self, limit=0) -> List[Dict[str, any]]:
        df = pd.read_json(self.f_name, lines=True)
        if limit > 0:
            df = df[:limit]
        df = self.add_col_alias(df)
        return df.to_dict(orient="records")

    def load_slice(self, offset=0, limit=0) -> List[Dict[str, any]]:
        if offset < 0 or limit < 0:
            raise ValueError("offset and limit must be non-negative")
        df = pd.read_json(self.f_name, lines=True)
        end = offset + limit if limit else None
        df = df.iloc[offset:end]
        df = self.add_col_alias(df)
        return df.to_dict(orient="records")


class RelativePath(JsonlFile):
    def __init__(
        self,
        f_name: str,
        default_task: str,
        ref_col: str,
        file_path_prefix: str,
        col_aliases=None,
    ):
        super().__init__(f_name, default_task, ref_col, col_aliases)
        if not file_path_prefix.endswith("/"):
            file_path_prefix += "/"
        self.file_path = file_path_prefix

    def load(self, limit=0) -> List[Dict[str, any]]:
        df = pd.read_json(self.f_name, lines=True)
        if limit > 0:
            df = df[:limit]

        def abs_path(x):
            temp = os.path.join(self.file_path, str(x))
            if os.path.exists(temp) and os.path.isfile(temp):
                return temp
            return x

        for item in df.columns:
            df[item] = df[item].progress_apply(abs_path)
        df = self.add_col_alias(df)
        return df.to_dict(orient="records")

    def load_slice(self, offset=0, limit=0) -> List[Dict[str, any]]:
        if offset < 0 or limit < 0:
            raise ValueError("offset and limit must be non-negative")
        df = pd.read_json(self.f_name, lines=True)
        end = offset + limit if limit else None
        df = df.iloc[offset:end]

        def abs_path(x):
            temp = os.path.join(self.file_path, str(x))
            if os.path.exists(temp) and os.path.isfile(temp):
                return temp
            return x

        for item in df.columns:
            df[item] = df[item].progress_apply(abs_path)
        df = self.add_col_alias(df)
        return df.to_dict(orient="records")


class InMemoryDataset(Dataset):
    """A preloaded dataset used to keep GPU initialization after data loading."""

    def __init__(self, rows, default_task: str, ref_col: str, col_aliases=None):
        super().__init__(default_task, ref_col, col_aliases)
        self.rows = list(rows)

    @classmethod
    def from_dataset(cls, dataset: Dataset, rows):
        return cls(
            rows=rows,
            default_task=dataset.task_name,
            ref_col=dataset.ref_col,
            col_aliases=dataset.col_aliases,
        )

    def load(self, limit=0) -> List[Dict[str, any]]:
        return list(self.rows[:limit] if limit else self.rows)

    def load_slice(self, offset=0, limit=0) -> List[Dict[str, any]]:
        if offset < 0 or limit < 0:
            raise ValueError("offset and limit must be non-negative")
        end = offset + limit if limit else None
        return list(self.rows[offset:end])
