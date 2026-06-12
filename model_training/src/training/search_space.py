"""
Optuna hyperparameter search space definition.

Edit sample_hyperparameters() to change which hyperparameters are tuned and their ranges.
The returned dict is passed as the `override` argument to TrainerConfig.get_predefined_config().
"""

from optuna import Trial


def sample_hyperparameters(trial: Trial) -> dict:
    """
    Sample hyperparameters for one Optuna trial.

    Returns a dict suitable as the `override` arg to
    TrainerConfig.get_predefined_config(keyword, override=...).
    Only includes hyperparameter fields — base config identity fields
    (backbone, keyword, save_path, metadata_frames, etc.) are left to the base config.
    """
    params = {}

    # Core training
    params["lr"] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    params["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    params["batch_size"] = trial.suggest_categorical("batch_size", [8, 16, 32])

    # LR scheduler — type choice drives conditional args
    scheduler_name = trial.suggest_categorical(
        "lr_scheduler",
        ["REDUCE_ON_PLATEAU", "COSINE_WARM_RESTARTS", "COSINE_ANNEALING", "STEP_LR"],
    )
    params["lr_scheduler"] = scheduler_name

    scheduler_args = {}
    if scheduler_name == "COSINE_WARM_RESTARTS":
        scheduler_args["T_0"] = trial.suggest_int("cosine_T_0", 5, 30)
        scheduler_args["T_mult"] = trial.suggest_categorical("cosine_T_mult", [1, 2])
    elif scheduler_name == "COSINE_ANNEALING":
        scheduler_args["T_max"] = trial.suggest_int("cosine_T_max", 20, 100)
    elif scheduler_name == "STEP_LR":
        scheduler_args["step_size"] = trial.suggest_int("step_lr_step_size", 5, 30)
        scheduler_args["gamma"] = trial.suggest_float("step_lr_gamma", 0.1, 0.9)
    elif scheduler_name == "REDUCE_ON_PLATEAU":
        scheduler_args["patience"] = trial.suggest_int("plateau_patience", 5, 20)
    params["lr_scheduler_args"] = scheduler_args

    return params
