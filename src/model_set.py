import parse
import pickle
import joblib
import pathlib
import itertools

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


def batched(iterable, n):
    """Batch data into iterators of length n. The last batch may be shorter.
    https://stackoverflow.com/a/8998040
    """
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while True:
        chunk_it = itertools.islice(it, n)
        try:
            first_el = next(chunk_it)
        except StopIteration:
            return
        yield list(itertools.chain((first_el,), chunk_it))


class ModelSet:
    """
    A set of models of the same family trained on the same dataset with different seeds.
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

    def wrapped_train_func(self, train_func, seed, eval_func=None, eval_label="eval"):
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

            if isinstance(eval_func, dict):
                for metric_name, metric_func in eval_func.items():
                    metric_vals = metric_func(model)
                    run.log({metric_name: metric_vals})
            elif hasattr(eval_func, "__call__"):
                metric_vals = eval_func(model)
                run.log({eval_label: metric_vals})

            if self.log_artifact:
                artifact = wandb.Artifact(name=model_name, type="model")
                artifact.add_file(f.name)
                run.log_artifact(artifact)

    def wrapped_apply_func(self, apply_func, seed):
        with open(self.get_model_filename(seed), "rb") as f:
            model = self.serializer.load(f)
            return apply_func(model)

    def train_until_condition(
        self,
        train_func,
        condition_func,
        target_num_seeds,
        eval_func=None,
        max_seeds=1000,
        overwrite=False,
        verbose=False,
        n_jobs=4,
        **parallel_kwargs,
    ):
        """
        Train with different seeds until a given number of models satisfy a condition.

        Args:
          train_func: Function which takes as input a seed and returns a model.
          condition_func: Function for computing whether a model satisfies the condition.
          target_num_seeds: Target number of models satisfying the condition.
          eval_func: Function for computing model metrics for logging.
          overwrite: Whether to train again if a model file for a given seed exists.
          n_jobs: Number of jobs to execute in parallel.
        """
        self.model_path.mkdir(exist_ok=True)
        full_seed_range = range(max_seeds)

        seed_range = []
        existing_seed_range = []
        for seed in full_seed_range:
            filename = self.get_model_filename(seed)
            if not filename.exists() or overwrite:
                seed_range.append(seed)
            if filename.exists():
                existing_seed_range.append(seed)

        def count_condition(seeds):
            return sum(
                int(flag)
                for flag in self.apply(
                    condition_func, seeds=seeds, n_jobs=n_jobs, **parallel_kwargs
                )
            )

        num_condition_models = count_condition(existing_seed_range)
        with joblib.Parallel(n_jobs=n_jobs) as parallel:
            it = batched(seed_range, n_jobs)
            if verbose:
                pbar = tqdm.tqdm(total=target_num_seeds)
                pbar.update(num_condition_models)
            for seed_batch in it:
                parallel(
                    joblib.delayed(self.wrapped_train_func)(
                        seed=seed, train_func=train_func, eval_func=eval_func
                    )
                    for seed in seed_batch
                )
                new_counts = count_condition(seed_batch)
                num_condition_models += new_counts
                if verbose:
                    pbar.update(new_counts)
                if num_condition_models >= target_num_seeds:
                    break

    def train(
        self,
        train_func,
        seeds,
        eval_func=None,
        criterion_func=None,
        overwrite=False,
        verbose=False,
        n_jobs=4,
        **parallel_kwargs,
    ):
        """
        Train several models with different seeds.

        Args:
          train_func: Function which takes as input a seed and returns a model.
          seeds: Either a list of seeds of a number of seeds to train.
          eval_func: Function for computing model metrics for logging.
          criterion_func: Function for computing whether a model fits a criterion.
            If such function is provided and the seeds value is a number, we will
            train until 'seeds' models satisfy the criterion. This turns the training
            into a kind of reservoir sampling.
          n_jobs: Number of jobs to execute in parallel.
          overwrite: Whether to train again if a model file for a given seed exists.
        """
        self.model_path.mkdir(exist_ok=True)
        if isinstance(seeds, int):
            seeds = range(seeds)

        seeds_to_train = []
        for seed in seeds:
            if (not self.get_model_filename(seed).exists()) or overwrite:
                seeds_to_train.append(seed)

        if verbose:
            seeds_to_train = tqdm.tqdm(seeds_to_train)

        tasks = (
            joblib.delayed(self.wrapped_train_func)(
                seed=seed, train_func=train_func, eval_func=eval_func
            )
            for seed in seeds_to_train
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

    def apply(self, apply_func, seeds=None, verbose=False, n_jobs=4, **parallel_kwargs):
        if seeds is None:
            seeds = self.get_seeds()

        for seed in seeds:
            filename = self.get_model_filename(seed)
            if not filename.exists():
                raise ValueError(f"Model {seed} not found: {filename}")

        if verbose:
            seeds = tqdm.tqdm(seeds)

        tasks = (
            joblib.delayed(self.wrapped_apply_func)(apply_func, seed) for seed in seeds
        )
        return joblib.Parallel(n_jobs=n_jobs, **parallel_kwargs)(tasks)
