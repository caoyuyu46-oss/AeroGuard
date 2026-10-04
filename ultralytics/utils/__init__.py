# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import importlib.metadata
import inspect
import json
import logging.config
import os
import platform
import re
import subprocess
import sys
import threading
import time
import urllib
import uuid
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm as tqdm_original

from ultralytics import __version__


RANK = int(os.getenv("RANK", -1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  


ARGV = sys.argv or ["", ""]  
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  
ASSETS = ROOT / "assets"  
DEFAULT_CFG_PATH = ROOT / "cfg/default.yaml"
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  
AUTOINSTALL = str(os.getenv("YOLO_AUTOINSTALL", True)).lower() == "true"  
VERBOSE = str(os.getenv("YOLO_VERBOSE", True)).lower() == "true"  
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}" if VERBOSE else None  
LOGGING_NAME = "ultralytics"
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])  
ARM64 = platform.machine() in {"arm64", "aarch64"}  
PYTHON_VERSION = platform.python_version()
TORCH_VERSION = torch.__version__
TORCHVISION_VERSION = importlib.metadata.version("torchvision")  
IS_VSCODE = os.environ.get("TERM_PROGRAM", False) == "vscode"
HELP_MSG = """
    Examples for running Ultralytics:

    1. Install the ultralytics package:

        pip install ultralytics

    2. Use the Python SDK:

        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n.yaml")  # build a new model from scratch
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

        # Use the model
        results = model.train(data="coco8.yaml", epochs=3)  # train the model
        results = model.val()  # evaluate model performance on the validation set
        results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
        success = model.export(format="onnx")  # export the model to ONNX format

    3. Use the command line interface (CLI):

        Ultralytics 'yolo' CLI commands use the following syntax:

            yolo TASK MODE ARGS

            Where   TASK (optional) is one of [detect, segment, classify, pose, obb]
                    MODE (required) is one of [train, val, predict, export, benchmark]
                    ARGS (optional) are any number of custom "arg=value" pairs like "imgsz=320" that override defaults.
                        See all ARGS at https://docs.ultralytics.com/usage/cfg or with "yolo cfg"

        - Train a detection model for 10 epochs with an initial learning_rate of 0.01
            yolo detect train data=coco8.yaml model=yolov8n.pt epochs=10 lr0=0.01

        - Predict a YouTube video using a pretrained segmentation model at image size 320:
            yolo segment predict model=yolov8n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320

        - Val a pretrained detection model at batch-size 1 and image size 640:
            yolo detect val model=yolov8n.pt data=coco8.yaml batch=1 imgsz=640

        - Export a YOLOv8n classification model to ONNX format at image size 224 by 128 (no TASK required)
            yolo export model=yolov8n-cls.pt format=onnx imgsz=224,128

        - Run special commands:
            yolo help
            yolo checks
            yolo version
            yolo settings
            yolo copy-cfg
            yolo cfg

    Docs: https://docs.ultralytics.com
    Community: https://community.ultralytics.com
    GitHub: https://github.com/ultralytics/ultralytics
    """


torch.set_printoptions(linewidth=320, precision=4, profile="default")
np.set_printoptions(linewidth=320, formatter={"float_kind": "{:11.5g}".format})  
cv2.setNumThreads(0)  
os.environ["NUMEXPR_MAX_THREADS"] = str(NUM_THREADS)  
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"  
os.environ["KINETO_LOG_LEVEL"] = "5"  


class TQDM(tqdm_original):
    

    def __init__(self, *args, **kwargs):
        
        kwargs["disable"] = not VERBOSE or kwargs.get("disable", False)  
        kwargs.setdefault("bar_format", TQDM_BAR_FORMAT)  
        super().__init__(*args, **kwargs)


class SimpleClass:
    

    def __str__(self):
        
        attr = []
        for a in dir(self):
            v = getattr(self, a)
            if not callable(v) and not a.startswith("_"):
                if isinstance(v, SimpleClass):
                    
                    s = f"{a}: {v.__module__}.{v.__class__.__name__} object"
                else:
                    s = f"{a}: {repr(v)}"
                attr.append(s)
        return f"{self.__module__}.{self.__class__.__name__} object with attributes:\n\n" + "\n".join(attr)

    def __repr__(self):
        
        return self.__str__()

    def __getattr__(self, attr):
        
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")


class IterableSimpleNamespace(SimpleNamespace):
    

    def __iter__(self):
        
        return iter(vars(self).items())

    def __str__(self):
        
        return "\n".join(f"{k}={v}" for k, v in vars(self).items())

    def __getattr__(self, attr):
        
        name = self.__class__.__name__
        raise AttributeError(
            f"""
            '{name}' object has no attribute '{attr}'. This may be caused by a modified or out of date ultralytics
            'default.yaml' file.\nPlease update your code with 'pip install -U ultralytics' and if necessary replace
            {DEFAULT_CFG_PATH} with the latest version from
            https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
            """
        )

    def get(self, key, default=None):
        
        return getattr(self, key, default)


def plt_settings(rcparams=None, backend="Agg"):
    
    if rcparams is None:
        rcparams = {"font.size": 11}

    def decorator(func):
        

        def wrapper(*args, **kwargs):
            
            original_backend = plt.get_backend()
            switch = backend.lower() != original_backend.lower()
            if switch:
                plt.close("all")  
                plt.switch_backend(backend)

            
            try:
                with plt.rc_context(rcparams):
                    result = func(*args, **kwargs)
            finally:
                if switch:
                    plt.close("all")
                    plt.switch_backend(original_backend)
            return result

        return wrapper

    return decorator


def set_logging(name="LOGGING_NAME", verbose=True):
    
    level = logging.INFO if verbose and RANK in {-1, 0} else logging.ERROR  

    
    formatter = logging.Formatter("%(message)s")  
    if WINDOWS and hasattr(sys.stdout, "encoding") and sys.stdout.encoding != "utf-8":

        class CustomFormatter(logging.Formatter):
            def format(self, record):
                
                return emojis(super().format(record))

        try:
            
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            
            elif hasattr(sys.stdout, "buffer"):
                import io

                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
            else:
                formatter = CustomFormatter("%(message)s")
        except Exception as e:
            print(f"Creating custom formatter for non UTF-8 environments due to {e}")
            formatter = CustomFormatter("%(message)s")

    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger



LOGGER = set_logging(LOGGING_NAME, verbose=VERBOSE)  
for logger in "sentry_sdk", "urllib3.connectionpool":
    logging.getLogger(logger).setLevel(logging.CRITICAL + 1)


def emojis(string=""):
    
    return string.encode().decode("ascii", "ignore") if WINDOWS else string


class ThreadingLocked:
    

    def __init__(self):
        
        self.lock = threading.Lock()

    def __call__(self, f):
        
        from functools import wraps

        @wraps(f)
        def decorated(*args, **kwargs):
            
            with self.lock:
                return f(*args, **kwargs)

        return decorated


def yaml_save(file="data.yaml", data=None, header=""):
    
    if data is None:
        data = {}
    file = Path(file)
    if not file.parent.exists():
        
        file.parent.mkdir(parents=True, exist_ok=True)

    
    valid_types = int, float, str, bool, list, tuple, dict, type(None)
    for k, v in data.items():
        if not isinstance(v, valid_types):
            data[k] = str(v)

    
    with open(file, "w", errors="ignore", encoding="utf-8") as f:
        if header:
            f.write(header)
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def yaml_load(file="data.yaml", append_filename=False):
    
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  

        
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

        
        data = yaml.safe_load(s) or {}  
        if append_filename:
            data["yaml_file"] = str(file)
        return data


def yaml_print(yaml_file: Union[str, Path, dict]) -> None:
    
    yaml_dict = yaml_load(yaml_file) if isinstance(yaml_file, (str, Path)) else yaml_file
    dump = yaml.dump(yaml_dict, sort_keys=False, allow_unicode=True, width=float("inf"))
    LOGGER.info(f"Printing '{colorstr('bold', 'black', yaml_file)}'\n\n{dump}")



DEFAULT_CFG_DICT = yaml_load(DEFAULT_CFG_PATH)
for k, v in DEFAULT_CFG_DICT.items():
    if isinstance(v, str) and v.lower() == "none":
        DEFAULT_CFG_DICT[k] = None
DEFAULT_CFG_KEYS = DEFAULT_CFG_DICT.keys()
DEFAULT_CFG = IterableSimpleNamespace(**DEFAULT_CFG_DICT)


def read_device_model() -> str:
    
    with contextlib.suppress(Exception):
        with open("/proc/device-tree/model") as f:
            return f.read()
    return ""


def is_ubuntu() -> bool:
    
    with contextlib.suppress(FileNotFoundError):
        with open("/etc/os-release") as f:
            return "ID=ubuntu" in f.read()
    return False


def is_colab():
    
    return "COLAB_RELEASE_TAG" in os.environ or "COLAB_BACKEND_VERSION" in os.environ


def is_kaggle():
    
    return os.environ.get("PWD") == "/kaggle/working" and os.environ.get("KAGGLE_URL_BASE") == "https://www.kaggle.com"


def is_jupyter():
    
    with contextlib.suppress(Exception):
        from IPython import get_ipython

        return get_ipython() is not None
    return False


def is_docker() -> bool:
    
    with contextlib.suppress(Exception):
        with open("/proc/self/cgroup") as f:
            return "docker" in f.read()
    return False


def is_raspberrypi() -> bool:
    
    return "Raspberry Pi" in PROC_DEVICE_MODEL


def is_jetson() -> bool:
    
    return "NVIDIA" in PROC_DEVICE_MODEL  


def is_online() -> bool:
    
    with contextlib.suppress(Exception):
        assert str(os.getenv("YOLO_OFFLINE", "")).lower() != "true"  
        import socket

        for dns in ("1.1.1.1", "8.8.8.8"):  
            socket.create_connection(address=(dns, 80), timeout=2.0).close()
            return True
    return False


def is_pip_package(filepath: str = __name__) -> bool:
    
    import importlib.util

    
    spec = importlib.util.find_spec(filepath)

    
    return spec is not None and spec.origin is not None


def is_dir_writeable(dir_path: Union[str, Path]) -> bool:
    
    return os.access(str(dir_path), os.W_OK)


def is_pytest_running():
    
    return ("PYTEST_CURRENT_TEST" in os.environ) or ("pytest" in sys.modules) or ("pytest" in Path(ARGV[0]).stem)


def is_github_action_running() -> bool:
    
    return "GITHUB_ACTIONS" in os.environ and "GITHUB_WORKFLOW" in os.environ and "RUNNER_OS" in os.environ


def get_git_dir():
    
    for d in Path(__file__).parents:
        if (d / ".git").is_dir():
            return d


def is_git_dir():
    
    return GIT_DIR is not None


def get_git_origin_url():
    
    if IS_GIT_DIR:
        with contextlib.suppress(subprocess.CalledProcessError):
            origin = subprocess.check_output(["git", "config", "--get", "remote.origin.url"])
            return origin.decode().strip()


def get_git_branch():
    
    if IS_GIT_DIR:
        with contextlib.suppress(subprocess.CalledProcessError):
            origin = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            return origin.decode().strip()


def get_default_args(func):
    
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def get_ubuntu_version():
    
    if is_ubuntu():
        with contextlib.suppress(FileNotFoundError, AttributeError):
            with open("/etc/os-release") as f:
                return re.search(r'VERSION_ID="(\d+\.\d+)"', f.read())[1]


def get_user_config_dir(sub_dir="Ultralytics"):
    
    if WINDOWS:
        path = Path.home() / "AppData" / "Roaming" / sub_dir
    elif MACOS:  
        path = Path.home() / "Library" / "Application Support" / sub_dir
    elif LINUX:
        path = Path.home() / ".config" / sub_dir
    else:
        raise ValueError(f"Unsupported operating system: {platform.system()}")

    
    if not is_dir_writeable(path.parent):
        LOGGER.warning(
            f"WARNING ⚠️ user config directory '{path}' is not writeable, defaulting to '/tmp' or CWD."
            "Alternatively you can define a YOLO_CONFIG_DIR environment variable for this path."
        )
        path = Path("/tmp") / sub_dir if is_dir_writeable("/tmp") else Path().cwd() / sub_dir

    
    path.mkdir(parents=True, exist_ok=True)

    return path



PROC_DEVICE_MODEL = read_device_model()  
ONLINE = is_online()
IS_COLAB = is_colab()
IS_DOCKER = is_docker()
IS_JETSON = is_jetson()
IS_JUPYTER = is_jupyter()
IS_KAGGLE = is_kaggle()
IS_PIP_PACKAGE = is_pip_package()
IS_RASPBERRYPI = is_raspberrypi()
GIT_DIR = get_git_dir()
IS_GIT_DIR = is_git_dir()
USER_CONFIG_DIR = Path(os.getenv("YOLO_CONFIG_DIR") or get_user_config_dir())  
SETTINGS_FILE = USER_CONFIG_DIR / "settings.json"


def colorstr(*input):
    
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  
    colors = {
        "black": "\033[30m",  
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def remove_colorstr(input_string):
    
    ansi_escape = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")
    return ansi_escape.sub("", input_string)


class TryExcept(contextlib.ContextDecorator):
    

    def __init__(self, msg="", verbose=True):
        
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        
        pass

    def __exit__(self, exc_type, value, traceback):
        
        if self.verbose and value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True


class Retry(contextlib.ContextDecorator):
    

    def __init__(self, times=3, delay=2):
        
        self.times = times
        self.delay = delay
        self._attempts = 0

    def __call__(self, func):
        

        def wrapped_func(*args, **kwargs):
            
            self._attempts = 0
            while self._attempts < self.times:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self._attempts += 1
                    print(f"Retry {self._attempts}/{self.times} failed: {e}")
                    if self._attempts >= self.times:
                        raise e
                    time.sleep(self.delay * (2**self._attempts))  

        return wrapped_func


def threaded(func):
    

    def wrapper(*args, **kwargs):
        
        if kwargs.pop("threaded", True):  
            thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
            thread.start()
            return thread
        else:
            return func(*args, **kwargs)

    return wrapper


def set_sentry():
    

    def before_send(event, hint):
        
        if "exc_info" in hint:
            exc_type, exc_value, tb = hint["exc_info"]
            if exc_type in {KeyboardInterrupt, FileNotFoundError} or "out of memory" in str(exc_value):
                return None  

        event["tags"] = {
            "sys_argv": ARGV[0],
            "sys_argv_name": Path(ARGV[0]).name,
            "install": "git" if IS_GIT_DIR else "pip" if IS_PIP_PACKAGE else "other",
            "os": ENVIRONMENT,
        }
        return event

    if (
        SETTINGS["sync"]
        and RANK in {-1, 0}
        and Path(ARGV[0]).name == "yolo"
        and not TESTS_RUNNING
        and ONLINE
        and IS_PIP_PACKAGE
        and not IS_GIT_DIR
    ):
        
        try:
            import sentry_sdk  
        except ImportError:
            return

        sentry_sdk.init(
            dsn="https://5ff1556b71594bfea135ff0203a0d290@o4504521589325824.ingest.sentry.io/4504521592406016",
            debug=False,
            traces_sample_rate=1.0,
            release=__version__,
            environment="production",  
            before_send=before_send,
            ignore_errors=[KeyboardInterrupt, FileNotFoundError],
        )
        sentry_sdk.set_user({"id": SETTINGS["uuid"]})  


class JSONDict(dict):
    

    def __init__(self, file_path: Union[str, Path] = "data.json"):
        
        super().__init__()
        self.file_path = Path(file_path)
        self.lock = Lock()
        self._load()

    def _load(self):
        
        try:
            if self.file_path.exists():
                with open(self.file_path) as f:
                    self.update(json.load(f))
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {self.file_path}. Starting with an empty dictionary.")
        except Exception as e:
            print(f"Error reading from {self.file_path}: {e}")

    def _save(self):
        
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, "w") as f:
                json.dump(dict(self), f, indent=2, default=self._json_default)
        except Exception as e:
            print(f"Error writing to {self.file_path}: {e}")

    @staticmethod
    def _json_default(obj):
        
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def __setitem__(self, key, value):
        
        with self.lock:
            super().__setitem__(key, value)
            self._save()

    def __delitem__(self, key):
        
        with self.lock:
            super().__delitem__(key)
            self._save()

    def __str__(self):
        
        return f'JSONDict("{self.file_path}"):\n{json.dumps(dict(self), indent=2, ensure_ascii=False, default=self._json_default)}'

    def update(self, *args, **kwargs):
        
        with self.lock:
            super().update(*args, **kwargs)
            self._save()

    def clear(self):
        
        with self.lock:
            super().clear()
            self._save()


class SettingsManager(JSONDict):
    

    def __init__(self, file=SETTINGS_FILE, version="0.0.6"):
        
        import hashlib

        from ultralytics.utils.torch_utils import torch_distributed_zero_first

        root = GIT_DIR or Path()
        datasets_root = (root.parent if GIT_DIR and is_dir_writeable(root.parent) else root).resolve()

        self.file = Path(file)
        self.version = version
        self.defaults = {
            "settings_version": version,
            "datasets_dir": str(datasets_root / "datasets"),
            "weights_dir": str(root / "weights"),
            "runs_dir": str(root / "runs"),
            "uuid": hashlib.sha256(str(uuid.getnode()).encode()).hexdigest(),
            "sync": True,
            "api_key": "",
            "openai_api_key": "",
            "clearml": True,  
            "comet": True,
            "dvc": True,
            "hub": True,
            "mlflow": True,
            "neptune": True,
            "raytune": True,
            "tensorboard": True,
            "wandb": True,
            "vscode_msg": True,
        }
        self.help_msg = (
            f"\nView Ultralytics Settings with 'yolo settings' or at '{self.file}'"
            "\nUpdate Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. "
            "For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings."
        )

        with torch_distributed_zero_first(RANK):
            super().__init__(self.file)

            if not self.file.exists() or not self:  
                LOGGER.info(f"Creating new Ultralytics Settings v{version} file ✅ {self.help_msg}")
                self.reset()

            self._validate_settings()

    def _validate_settings(self):
        
        correct_keys = set(self.keys()) == set(self.defaults.keys())
        correct_types = all(isinstance(self.get(k), type(v)) for k, v in self.defaults.items())
        correct_version = self.get("settings_version", "") == self.version

        if not (correct_keys and correct_types and correct_version):
            LOGGER.warning(
                "WARNING ⚠️ Ultralytics settings reset to default values. This may be due to a possible problem "
                f"with your settings or a recent ultralytics package update. {self.help_msg}"
            )
            self.reset()

        if self.get("datasets_dir") == self.get("runs_dir"):
            LOGGER.warning(
                f"WARNING ⚠️ Ultralytics setting 'datasets_dir: {self.get('datasets_dir')}' "
                f"must be different than 'runs_dir: {self.get('runs_dir')}'. "
                f"Please change one to avoid possible issues during training. {self.help_msg}"
            )

    def update(self, *args, **kwargs):
        
        for k, v in kwargs.items():
            if k not in self.defaults:
                raise KeyError(f"No Ultralytics setting '{k}'. {self.help_msg}")
            t = type(self.defaults[k])
            if not isinstance(v, t):
                raise TypeError(f"Ultralytics setting '{k}' must be of type '{t}', not '{type(v)}'. {self.help_msg}")
        super().update(*args, **kwargs)

    def reset(self):
        
        self.clear()
        self.update(self.defaults)


def deprecation_warn(arg, new_arg):
    
    LOGGER.warning(f"WARNING ⚠️ '{arg}' is deprecated and will be removed in in the future. Use '{new_arg}' instead.")


def clean_url(url):
    
    url = Path(url).as_posix().replace(":/", "://")  
    return urllib.parse.unquote(url).split("?")[0]  


def url2file(url):
    
    return Path(clean_url(url)).name


def vscode_msg(ext="ultralytics.ultralytics-snippets") -> str:
    
    path = (USER_CONFIG_DIR.parents[2] if WINDOWS else USER_CONFIG_DIR.parents[1]) / ".vscode/extensions"
    obs_file = path / ".obsolete"  
    installed = any(path.glob(f"{ext}*")) and ext not in (obs_file.read_text("utf-8") if obs_file.exists() else "")
    url = "https://docs.ultralytics.com/integrations/vscode"
    return "" if installed else f"{colorstr('VS Code:')} view Ultralytics VS Code Extension ⚡ at {url}"





PREFIX = colorstr("Ultralytics: ")
SETTINGS = SettingsManager()  
PERSISTENT_CACHE = JSONDict(USER_CONFIG_DIR / "persistent_cache.json")  
DATASETS_DIR = Path(SETTINGS["datasets_dir"])  
WEIGHTS_DIR = Path(SETTINGS["weights_dir"])  
RUNS_DIR = Path(SETTINGS["runs_dir"])  
ENVIRONMENT = (
    "Colab"
    if IS_COLAB
    else "Kaggle"
    if IS_KAGGLE
    else "Jupyter"
    if IS_JUPYTER
    else "Docker"
    if IS_DOCKER
    else platform.system()
)
TESTS_RUNNING = is_pytest_running() or is_github_action_running()
set_sentry()


from ultralytics.utils.patches import imread, imshow, imwrite, torch_load, torch_save

torch.load = torch_load
torch.save = torch_save
if WINDOWS:
    
    cv2.imread, cv2.imwrite, cv2.imshow = imread, imwrite, imshow
