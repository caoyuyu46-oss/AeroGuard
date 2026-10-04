# Ultralytics YOLO 🚀, AGPL-3.0 license

import subprocess

from ultralytics.cfg import TASK2DATA, TASK2METRIC, get_save_dir
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT, LOGGER, NUM_THREADS, checks


def run_ray_tune(
    model, space: dict = None, grace_period: int = 10, gpu_per_trial: int = None, max_samples: int = 10, **train_args
):
    
    LOGGER.info("💡 Learn about RayTune at https://docs.ultralytics.com/integrations/ray-tune")
    if train_args is None:
        train_args = {}

    try:
        subprocess.run("pip install ray[tune]".split(), check=True)  

        import ray
        from ray import tune
        from ray.air import RunConfig
        from ray.air.integrations.wandb import WandbLoggerCallback
        from ray.tune.schedulers import ASHAScheduler
    except ImportError:
        raise ModuleNotFoundError('Ray Tune required but not found. To install run: pip install "ray[tune]"')

    try:
        import wandb

        assert hasattr(wandb, "__version__")
    except (ImportError, AssertionError):
        wandb = False

    checks.check_version(ray.__version__, ">=2.0.0", "ray")
    default_space = {
        
        "lr0": tune.uniform(1e-5, 1e-1),
        "lrf": tune.uniform(0.01, 1.0),  
        "momentum": tune.uniform(0.6, 0.98),  
        "weight_decay": tune.uniform(0.0, 0.001),  
        "warmup_epochs": tune.uniform(0.0, 5.0),  
        "warmup_momentum": tune.uniform(0.0, 0.95),  
        "box": tune.uniform(0.02, 0.2),  
        "cls": tune.uniform(0.2, 4.0),  
        "hsv_h": tune.uniform(0.0, 0.1),  
        "hsv_s": tune.uniform(0.0, 0.9),  
        "hsv_v": tune.uniform(0.0, 0.9),  
        "degrees": tune.uniform(0.0, 45.0),  
        "translate": tune.uniform(0.0, 0.9),  
        "scale": tune.uniform(0.0, 0.9),  
        "shear": tune.uniform(0.0, 10.0),  
        "perspective": tune.uniform(0.0, 0.001),  
        "flipud": tune.uniform(0.0, 1.0),  
        "fliplr": tune.uniform(0.0, 1.0),  
        "bgr": tune.uniform(0.0, 1.0),  
        "mosaic": tune.uniform(0.0, 1.0),  
        "mixup": tune.uniform(0.0, 1.0),  
        "copy_paste": tune.uniform(0.0, 1.0),  
    }

    
    task = model.task
    model_in_store = ray.put(model)

    def _tune(config):
        
        model_to_train = ray.get(model_in_store)  
        model_to_train.reset_callbacks()
        config.update(train_args)
        results = model_to_train.train(**config)
        return results.results_dict

    
    if not space:
        space = default_space
        LOGGER.warning("WARNING ⚠️ search space not provided, using default search space.")

    
    data = train_args.get("data", TASK2DATA[task])
    space["data"] = data
    if "data" not in train_args:
        LOGGER.warning(f'WARNING ⚠️ data not provided, using default "data={data}".')

    
    trainable_with_resources = tune.with_resources(_tune, {"cpu": NUM_THREADS, "gpu": gpu_per_trial or 0})

    
    asha_scheduler = ASHAScheduler(
        time_attr="epoch",
        metric=TASK2METRIC[task],
        mode="max",
        max_t=train_args.get("epochs") or DEFAULT_CFG_DICT["epochs"] or 100,
        grace_period=grace_period,
        reduction_factor=3,
    )

    
    tuner_callbacks = [WandbLoggerCallback(project="YOLOv8-tune")] if wandb else []

    
    tune_dir = get_save_dir(DEFAULT_CFG, name="tune").resolve()  
    tune_dir.mkdir(parents=True, exist_ok=True)
    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=space,
        tune_config=tune.TuneConfig(scheduler=asha_scheduler, num_samples=max_samples),
        run_config=RunConfig(callbacks=tuner_callbacks, storage_path=tune_dir),
    )

    
    tuner.fit()

    
    results = tuner.get_results()

    
    ray.shutdown()

    return results
