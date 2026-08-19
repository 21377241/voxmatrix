import atexit
import hashlib
import json
import os
import shlex
import signal
import subprocess
import logging
import tempfile
import threading
from contextlib import contextmanager
from functools import wraps

logger = logging.getLogger(__name__)

_ENV_PREPARE_VERSION = "isolated-env-v1"
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_EVALUATION_UTILS_ROOT = os.path.join(
    os.path.dirname(_REPO_ROOT), "evaluation_utils"
)
_ENV_THREAD_LOCKS = {}
_ENV_THREAD_LOCKS_GUARD = threading.Lock()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, payload):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


@contextmanager
def _environment_lock(path):
    with _ENV_THREAD_LOCKS_GUARD:
        thread_lock = _ENV_THREAD_LOCKS.setdefault(path, threading.Lock())
    with thread_lock:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        handle = open(path, "a+", encoding="utf-8")
        try:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except ImportError:
                logger.warning("fcntl is unavailable; isolated env lock is process-local")
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except ImportError:
                pass
            handle.close()


def _prepare_isolated_environment(env_path, requirements_path, pre_command=""):
    """Create/install one isolated env exactly once for concurrent replicas."""
    raw_env_path = str(env_path or "").rstrip("/")
    if not raw_env_path:
        raise ValueError("env_path cannot be empty")
    env_path = os.path.abspath(raw_env_path)
    requirements_path = (
        os.path.abspath(requirements_path) if requirements_path else ""
    )
    if requirements_path and not os.path.isfile(requirements_path):
        raise FileNotFoundError(
            f"isolated requirements file does not exist: {requirements_path}"
        )
    identity = {
        "format_version": _ENV_PREPARE_VERSION,
        "python": "3.10",
        "requirements_path": requirements_path,
        "requirements_sha256": (
            _sha256_file(requirements_path) if requirements_path else ""
        ),
        "pre_command": pre_command or "",
    }
    lock_path = f"{env_path}.ultraeval-prepare.lock"
    sentinel_path = os.path.join(env_path, ".ultraeval-prepare.json")
    with _environment_lock(lock_path):
        python_exe = os.path.join(env_path, "bin", "python")
        if requirements_path and os.path.isfile(python_exe) and os.path.isfile(
            sentinel_path
        ):
            try:
                with open(sentinel_path, encoding="utf-8") as handle:
                    if json.load(handle) == identity:
                        logger.info("Reusing prepared isolated environment: %s", env_path)
                        return python_exe
            except (OSError, ValueError, json.JSONDecodeError):
                pass

        if not os.path.isfile(python_exe):
            result = subprocess.run(
                ["uv", "venv", env_path, "--python", "3.10", "--allow-existing"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "Failed to create virtual environment: "
                    f"{result.stderr or result.stdout}"
                )
        if not os.path.isfile(python_exe):
            raise RuntimeError(f"Python executable not found: {python_exe}")

        if requirements_path:
            activate_path = os.path.join(env_path, "bin", "activate")
            if os.path.isfile(activate_path):
                activate_cmd = f"source {shlex.quote(activate_path)} && "
            else:
                activate_cmd = (
                    f"export PATH={shlex.quote(os.path.join(env_path, 'bin'))}:$PATH && "
                )
            commands = [
                'export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-auto}"',
            ]
            if pre_command:
                commands.append(pre_command)
            commands.extend(
                [
                    "uv pip install 'setuptools<81'",
                    "uv pip install --index-strategy unsafe-best-match "
                    f"-r {shlex.quote(requirements_path)}",
                ]
            )
            subprocess.run(
                activate_cmd + " && ".join(commands),
                shell=True,
                check=True,
                executable="/bin/bash",
            )
            _atomic_json(sentinel_path, identity)
        else:
            logger.info(
                "Skipping isolated dependency installation because requirements_path is empty."
            )
        return python_exe


def isolated(
    script_path: str, command_args_attr: str = "command_args", pre_command: str = ""
):
    def decorator(cls):
        original_init = cls.__init__

        @wraps(original_init)
        def new_init(self, env_path, requirements_path, *args, gpu_id=None, **kwargs):
            """
            Args:
                env_path: 虚拟环境路径
                requirements_path: 依赖文件路径
                gpu_id: 指定使用的 GPU ID，如 0, 1, 2。
                        如果为 None，则不设置 CUDA_VISIBLE_DEVICES（使用默认行为）
            """
            original_init(self, *args, **kwargs)
            raw_env_path = str(env_path or "").rstrip("/")
            if not raw_env_path:
                raise ValueError("env_path cannot be empty")
            env_path = os.path.abspath(raw_env_path)

            # 保存 gpu_id 供外部查询
            self._gpu_id = gpu_id

            python_exe = _prepare_isolated_environment(
                env_path, requirements_path, pre_command
            )
            if os.path.exists(f"{env_path}/bin/activate"):
                activate_cmd = f"source {shlex.quote(f'{env_path}/bin/activate')} && "
            else:
                activate_cmd = (
                    f"export PATH={shlex.quote(f'{env_path}/bin')}:$PATH && "
                )

            # 自动检测 Python 版本
            python_version = (
                subprocess.check_output(
                    f"{activate_cmd}{shlex.quote(python_exe)} --version",
                    shell=True,
                    executable="/bin/bash",
                    text=True,
                )
                .strip()
                .split()[1]
            )
            major_minor = ".".join(python_version.split(".")[:2])

            # 构建 LD_LIBRARY_PATH
            lib_path = (
                f"{env_path}/lib/python{major_minor}/site-packages/nvidia/nvjitlink/lib"
            )

            cuda_runtime_lib = f"{env_path}/lib/python{major_minor}/site-packages/nvidia/cuda_runtime/lib"

            evaluation_utils_root = os.environ.get(
                "EVALUATION_UTILS_ROOT",
                _DEFAULT_EVALUATION_UTILS_ROOT,
            )
            bundled_tools_path = ":".join(
                [
                    f"{evaluation_utils_root}/tools/ffmpeg",
                    f"{evaluation_utils_root}/tools/java/current/bin",
                ]
            )

            # 构建命令行参数
            command_args = getattr(self, command_args_attr, {})
            args_str = " ".join(
                [
                    f"--{key}" if value == "" else f"--{key} {shlex.quote(str(value))}"
                    for key, value in command_args.items()
                ]
            )

            # 构建 CUDA_VISIBLE_DEVICES 设置
            cuda_env = ""
            if gpu_id is not None:
                cuda_env = (
                    f"export CUDA_VISIBLE_DEVICES={shlex.quote(str(gpu_id))} && "
                )
                logger.info(
                    f"Setting CUDA_VISIBLE_DEVICES={gpu_id} for isolated process"
                )

            # 构建完整命令
            command = (
                activate_cmd
                + f"{cuda_env}"
                + f"export PATH={shlex.quote(bundled_tools_path)}:$PATH && "
                + "export LD_LIBRARY_PATH="
                + f"{shlex.quote(f'{lib_path}:{cuda_runtime_lib}')}:$LD_LIBRARY_PATH && "
                + f"{shlex.quote(python_exe)} -u {shlex.quote(script_path)} {args_str}"
            )
            logger.info(f"Running command: {command}")
            self.process = subprocess.Popen(
                command,
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                executable="/bin/bash",
                start_new_session=True,
            )

            def terminate_process_group(timeout=30):
                if self.process.poll() is not None:
                    return
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                except ProcessLookupError:
                    return
                except OSError:
                    self.process.terminate()
                try:
                    self.process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        self.process.kill()
                    self.process.wait(timeout=timeout)

            self._terminate_isolated_process = terminate_process_group

            # 添加检查进程状态并打印错误信息的方法
            def check_process_status(self_ref):
                """检查进程状态，如果进程已退出则打印所有输出信息"""
                if self_ref.process.poll() is not None:
                    exit_code = self_ref.process.returncode
                    logger.error(f"Process has exited with code: {exit_code}")
                    try:
                        # 读取剩余的输出
                        stdout, stderr = self_ref.process.communicate(timeout=5)
                        if stdout:
                            logger.error(f"Process STDOUT:\n{stdout}")
                        if stderr:
                            logger.error(f"Process STDERR:\n{stderr}")
                    except Exception as e:
                        logger.error(f"Failed to read process output: {e}")
                    return False
                return True

            self.check_process_status = lambda: check_process_status(self)

            # 注册清理函数
            def cleanup():
                if self.process.poll() is None:
                    terminate_process_group()
                else:
                    # 进程已退出，打印输出信息
                    exit_code = self.process.returncode
                    logger.info(f"Process already exited with code: {exit_code}")
                    try:
                        stdout, stderr = self.process.communicate(timeout=5)
                        if stdout:
                            logger.info(f"Final STDOUT:\n{stdout}")
                        if stderr:
                            logger.error(f"Final STDERR:\n{stderr}")
                    except Exception as e:
                        logger.warning(f"Could not read final output: {e}")

            atexit.register(cleanup)
            self._isolated_cleanup = cleanup
            custom_wait_until_ready = getattr(self, "_wait_until_ready", None)
            self._isolated_ready = False
            self._isolated_ready_protocol = (
                "custom" if callable(custom_wait_until_ready) else "launch"
            )

            def wait_until_ready():
                if self._isolated_ready:
                    return
                if callable(custom_wait_until_ready):
                    custom_wait_until_ready()
                elif not self.check_process_status():
                    raise RuntimeError("isolated subprocess exited during launch")
                self._isolated_ready = True

            self.wait_until_ready = wait_until_ready
            try:
                wait_until_ready()
            except BaseException:
                terminate_process_group()
                raise

        cls.__init__ = new_init
        return cls

    return decorator
