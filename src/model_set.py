import parse
import pickle
import joblib
import pathlib

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
    model_filename_template = "model_{seed}"

    def __init__(
        self,
        model_path,
        serializer=None,
    ):
        self.model_path = model_path
        if serializer is None:
            serializer = PickleSerializer()
        self.serializer = serializer

    def get_model_filename(self, seed):
        return pathlib.Path(self.model_path) / ModelSet.model_filename_template.format(
            seed=seed
        )

    def wrapped_train_func(self, train_func, seed):
        with open(self.get_model_filename(seed), "wb") as f:
            model = train_func(seed)
            self.serializer.save(model, f)

    def wrapped_apply_func(self, apply_func, seed):
        with open(self.get_model_filename(seed), "rb") as f:
            model = self.serializer.load(f)
            return apply_func(model)

    def train(self, train_func, seeds, n_jobs=4, backend="loky"):
        if isinstance(seeds, int):
            seeds = range(seeds)

        tasks = (
            joblib.delayed(self.wrapped_train_func)(train_func, seed)
            for seed in tqdm.tqdm(seeds)
        )
        joblib.Parallel(n_jobs=n_jobs, backend=backend)(tasks)

    def get_seeds(self):
        seeds = []
        for filename in pathlib.Path(self.model_path).iterdir():
            res = parse.parse(self.model_filename_template, filename.name)
            try:
                seeds.append(int(res["seed"]))
            except KeyError:
                pass
        return seeds

    def apply(self, apply_func, n_jobs=4, backend="loky"):
        seeds = self.get_seeds()
        tasks = (
            joblib.delayed(self.wrapped_apply_func)(apply_func, seed)
            for seed in tqdm.tqdm(seeds)
        )
        return joblib.Parallel(n_jobs=n_jobs, backend=backend)(tasks)
