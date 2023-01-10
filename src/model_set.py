import parse
import pickle
import joblib
import pathlib

import wandb
from tqdm import autonotebook as tqdm

try:
    import dill
except ImportError:
    pass


class PickleSerializer:
    def save(self, obj, f):
        pickle.dump(obj, f)

    def load(self, f):
        return pickle.load(f)


class DillSerializer:
    def save(self, obj, f):
        dill.dump(obj, f)

    def load(self, f):
        return dill.load(f)


class ModelSet:
    """
    A set of models of the same architecture trained on the same dataset.
    """

    def __init__(
        self,
        model_path,
        model_filename_template="model_{seed}",
        serializer=None,
        run_params=None,
        log_artifact=None,
    ):
        self.model_path = pathlib.Path(model_path)
        self.model_filename_template = model_filename_template
        if serializer is None:
            serializer = PickleSerializer()
        self.serializer = serializer
        self.run_params = run_params or dict(mode="disabled")
        self.log_artifact = log_artifact

    def get_model_filename(self, seed):
        return self.model_path / self.model_filename_template.format(seed=seed)

    def wrapped_train_func(self, train_func, seed, eval_func=None):
        model_name = self.model_filename_template.format(seed=seed)
        with wandb.init(
            reinit=True,
            **self.run_params,
            name=model_name,
        ) as run:
            wandb.config.update(dict(seed=seed))
            model = train_func(seed)
            with open(self.get_model_filename(seed), "wb") as f:
                self.serializer.save(model, f)

            if eval_func is not None:
                eval_metric = eval_func(model)
                run.log({"eval": eval_metric})

            if self.log_artifact:
                artifact = wandb.Artifact(name=model_name, type="model")
                artifact.add_file(f.name)
                run.log_artifact(artifact)

    def wrapped_apply_func(self, apply_func, seed):
        with open(self.get_model_filename(seed), "rb") as f:
            model = self.serializer.load(f)
            return apply_func(model)

    def train(
        self,
        train_func,
        seeds,
        eval_func=None,
        n_jobs=4,
        overwrite=False,
        **parallel_kwargs
    ):
        self.model_path.mkdir(exist_ok=True)
        if isinstance(seeds, int):
            seeds = range(seeds)

        seeds_to_train = []
        for seed in seeds:
            if (not self.get_model_filename(seed).exists()) or overwrite:
                seeds_to_train.append(seed)

        tasks = (
            joblib.delayed(self.wrapped_train_func)(
                seed=seed, train_func=train_func, eval_func=eval_func
            )
            for seed in tqdm.tqdm(seeds_to_train)
        )
        joblib.Parallel(n_jobs=n_jobs, **parallel_kwargs)(tasks)

    def get_seeds(self):
        seeds = []
        for filename in pathlib.Path(self.model_path).iterdir():
            res = parse.parse(self.model_filename_template, filename.name)
            try:
                seeds.append(int(res["seed"]))
            except KeyError:
                pass
        return seeds

    def apply(self, apply_func, n_jobs=4, **parallel_kwargs):
        seeds = self.get_seeds()
        tasks = (
            joblib.delayed(self.wrapped_apply_func)(apply_func, seed)
            for seed in tqdm.tqdm(seeds)
        )
        return joblib.Parallel(n_jobs=n_jobs, **parallel_kwargs)(tasks)
