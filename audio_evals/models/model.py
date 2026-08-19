import logging
import os
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from typing import Dict

from audio_evals.base import PromptStruct
from audio_evals.utils import retry
from huggingface_hub import snapshot_download, HfApi
from audio_evals.constants import DEFAULT_MODEL_PATH

# the str type for pre-train model, the list type for chat model

logger = logging.getLogger(__name__)

_DOWNLOAD_LOCKS = {}
_DOWNLOAD_LOCKS_GUARD = threading.Lock()


@contextmanager
def _model_download_lock(local_dir):
    lock_path = f"{local_dir}.ultraeval-download.lock"
    with _DOWNLOAD_LOCKS_GUARD:
        thread_lock = _DOWNLOAD_LOCKS.setdefault(lock_path, threading.Lock())
    with thread_lock:
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        handle = open(lock_path, "a+", encoding="utf-8")
        try:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except ImportError:
                logger.warning("fcntl is unavailable; model download lock is process-local")
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except ImportError:
                pass
            handle.close()


def _all_expected_files_present(local_dir, expected_files):
    found_file = False
    for rel_path, remote_size in expected_files:
        if not rel_path:
            continue
        found_file = True
        target_path = os.path.abspath(os.path.join(local_dir, rel_path))
        root = os.path.abspath(local_dir) + os.sep
        if not target_path.startswith(root) or not os.path.isfile(target_path):
            return False
        if remote_size is not None:
            try:
                if os.path.getsize(target_path) != int(remote_size):
                    return False
            except (OSError, TypeError, ValueError):
                return False
    return found_file


class Model(ABC):
    def __init__(self, is_chat: bool, sample_params: Dict[str, any] = None):
        self.is_chat = is_chat
        if sample_params is None:
            sample_params = {}
        self.sample_params = sample_params

    @abstractmethod
    def _inference(self, prompt: PromptStruct, **kwargs):
        raise NotImplementedError()

    def inference(self, prompt: PromptStruct, **kwargs) -> str:
        if isinstance(prompt, list) and not self.is_chat:
            raise ValueError("struct input not match pre-train model")
        if isinstance(prompt, str) and self.is_chat:
            prompt = [{"role": "user", "contents": [{"type": "text", "value": prompt}]}]
        sample_params = deepcopy(self.sample_params)
        sample_params.update(kwargs)
        logger.debug(f"sample_params: {sample_params}\nprompt: {prompt}")
        return self._inference(prompt, **sample_params)


class OfflineModel(Model, ABC):

    def __init__(self, is_chat: bool, sample_params: Dict[str, any] = None):
        super().__init__(is_chat, sample_params)
        self.lock = threading.Lock()

    @staticmethod
    def _download_model(repo_id: str, repo_type: str = None) -> str:
        """Download model from HuggingFace Hub if not exists locally.

        Args:
            repo_id: HuggingFace repository ID (e.g. "openbmb/MiniCPM-o-2_6")

        Returns:
            str: Local path where model is downloaded
        """
        local_dir = os.path.join(DEFAULT_MODEL_PATH, repo_id)
        try:
            with _model_download_lock(local_dir):
                try:
                    if os.path.isdir(local_dir):
                        if os.environ.get("IGNORE_WEIGHT_CHECK", "") == "1":
                            return local_dir
                        info = HfApi().repo_info(
                            repo_id=repo_id, repo_type=repo_type, files_metadata=True
                        )
                        siblings = getattr(info, "siblings", []) or []
                        expected = [
                            (
                                getattr(sibling, "rfilename", None),
                                getattr(sibling, "size", None),
                            )
                            for sibling in siblings
                        ]
                        if _all_expected_files_present(local_dir, expected):
                            logger.info(
                                "Model already present locally, skip download: %s",
                                local_dir,
                            )
                            return local_dir
                except Exception as precheck_error:
                    logger.debug(
                        "Model pre-check failed, proceeding to download: %s",
                        precheck_error,
                        exc_info=True,
                    )

                logger.info("Downloading model from HuggingFace Hub: %s", repo_id)
                downloaded_dir = snapshot_download(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    local_dir=local_dir,
                    resume_download=True,
                    local_dir_use_symlinks=False,
                )
                logger.info("Model downloaded to: %s", downloaded_dir)
                return downloaded_dir
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return OfflineModel._download_model_from_modelscope(repo_id, repo_type)

    def inference(self, prompt: PromptStruct, **kwargs) -> str:
        with self.lock:
            return super().inference(prompt, **kwargs)

    @staticmethod
    def _download_model_from_modelscope(repo_id: str, repo_type: str = None) -> str:
        """
        从 ModelScope 平台下载模型（如果本地不存在的话）。

        Args:
            repo_id: ModelScope 平台的模型标识（例如 "damo/nlp_bert_base" 或类似）
            repo_type: 可选，类似 "model"、"dataset" 等

        Returns:
            str: 本地模型目录路径
        """
        del repo_type
        local_dir = os.path.join(DEFAULT_MODEL_PATH, repo_id)
        try:
            with _model_download_lock(local_dir):
                try:
                    if os.path.isdir(local_dir):
                        if os.environ.get("IGNORE_WEIGHT_CHECK", "") == "1":
                            return local_dir
                        from modelscope.hub.api import HubApi

                        files_info = HubApi().get_model_files(model_id=repo_id)
                        expected = [
                            (
                                item.get("Path") or item.get("Name"),
                                item.get("Size"),
                            )
                            for item in (files_info or [])
                        ]
                        if _all_expected_files_present(local_dir, expected):
                            logger.info(
                                "Model already present locally, skip download: %s",
                                local_dir,
                            )
                            return local_dir
                except Exception as precheck_error:
                    logger.debug(
                        "ModelScope pre-check failed, proceeding to download: %s",
                        precheck_error,
                        exc_info=True,
                    )

                logger.info("Downloading model from ModelScope: %s", repo_id)
                from modelscope.hub.snapshot_download import snapshot_download as ms_download

                model_dir = ms_download(repo_id, local_dir=local_dir)
                logger.info("Model downloaded to: %s", model_dir)
                return model_dir
        except Exception as e:
            raise RuntimeError(
                f"Failed to download model from ModelScope: {e}"
            ) from e


class APIModel(Model, ABC):

    @retry(max_retries=3)
    def inference(self, prompt: PromptStruct, **kwargs) -> str:
        return super().inference(prompt, **kwargs)
