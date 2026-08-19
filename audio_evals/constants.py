import os

from audio_evals.config import get_environment


DEFAULT_MODEL_PATH = os.path.abspath(
    os.path.expanduser(get_environment("VOXMATRIX_MODEL_ROOT", "init_model"))
)
