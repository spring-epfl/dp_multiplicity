#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Runs CIFAR10 training with differential privacy.
"""

import argparse
import logging
import os
import shutil
import sys
import pathlib
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from opacus import PrivacyEngine
from opacus.accountants import PRVAccountant
from opacus.validators import ModuleValidator
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.datasets import CIFAR10
from torchvision import models
from tqdm import tqdm

import wandb


logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("ddp")
logger.setLevel(level=logging.INFO)


def setup(args):
    if not torch.cuda.is_available():
        raise NotImplementedError(
            "DistributedDataParallel device_ids and output_device arguments \
            only work with single-device GPU modules"
        )

    if sys.platform == "win32":
        raise NotImplementedError("Windows version of multi-GPU is not supported yet.")

    # Initialize the process group on a Slurm cluster
    if os.environ.get("SLURM_NTASKS") is not None:
        rank = int(os.environ.get("SLURM_PROCID"))
        local_rank = int(os.environ.get("SLURM_LOCALID"))
        world_size = int(os.environ.get("SLURM_NTASKS"))
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "7440"

        torch.distributed.init_process_group(
            args.dist_backend, rank=rank, world_size=world_size
        )

        logger.debug(
            f"Setup on Slurm: rank={rank}, local_rank={local_rank}, world_size={world_size}"
        )

        return (rank, local_rank, world_size)

    # Initialize the process group through the environment variables
    elif args.local_rank >= 0:
        torch.distributed.init_process_group(
            init_method="env://",
            backend=args.dist_backend,
        )
        rank = torch.distributed.get_rank()
        local_rank = args.local_rank
        world_size = torch.distributed.get_world_size()

        logger.debug(
            f"Setup with 'env://': rank={rank}, local_rank={local_rank}, world_size={world_size}"
        )

        return (rank, local_rank, world_size)

    else:
        logger.debug(f"Running on a single GPU.")
        return (0, 0, 1)


def cleanup():
    torch.distributed.destroy_process_group()


def convnet(num_classes):
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(128, num_classes, bias=True),
    )


def save_checkpoint(state, is_best, filename="checkpoint.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f"{filename}_best")


def accuracy(preds, labels):
    return (preds == labels).mean()


def train(args, model, train_loader, optimizer, privacy_engine, epoch, device):
    start_time = datetime.now()

    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    top1_acc = []

    if args.grad_sample_mode == "no_op":
        from functorch import grad_and_value, make_functional, vmap

        # Functorch prepare
        fmodel, _fparams = make_functional(model)

        def compute_loss_stateless_model(params, sample, target):
            batch = sample.unsqueeze(0)
            targets = target.unsqueeze(0)

            predictions = fmodel(params, batch)
            loss = criterion(predictions, targets)
            return loss

        ft_compute_grad = grad_and_value(compute_loss_stateless_model)
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0, 0))
        # Using model.parameters() instead of fparams
        # as fparams seems to not point to the dynamically updated parameters
        params = list(model.parameters())

    for i, (images, target) in enumerate(train_loader):

        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)

        if args.grad_sample_mode == "no_op":
            per_sample_grads, per_sample_losses = ft_compute_sample_grad(
                params, images, target
            )
            per_sample_grads = [g.detach() for g in per_sample_grads]
            loss = torch.mean(per_sample_losses)
            for (p, g) in zip(params, per_sample_grads):
                p.grad_sample = g
        else:
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc1 = accuracy(preds, labels)
            top1_acc.append(acc1)

            # compute gradient and do SGD step
            loss.backward()

        losses.append(loss.item())

        # make sure we take a step after processing the last mini-batch in the
        # epoch to ensure we start the next epoch with a clean state
        optimizer.step()
        optimizer.zero_grad()

    train_duration = datetime.now() - start_time
    return train_duration, np.mean(losses)


def test(args, model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc1 = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc1)

    top1_avg = np.mean(top1_acc)

    # print(f"\tTest set:" f"loss: {np.mean(losses):.6f} " f"Acc@1: {top1_avg :.6f} ")
    return top1_avg, np.mean(losses)


# flake8: noqa: C901
def main():
    args = parse_args()

    if args.debug >= 1:
        logger.setLevel(level=logging.DEBUG)

    if args.sigma is None or args.sigma == "None":
        args.disable_dp = True
    else:
        args.sigma = float(args.sigma)

    acct = PRVAccountant()
    dataset_size = 50_000
    sample_rate = args.batch_size / dataset_size
    for i in tqdm(range(int(args.epochs * (1.0 / sample_rate)))):
        acct.step(noise_multiplier=args.sigma, sample_rate=sample_rate)
    print("Estimating privacy...")
    estimated_epsilon = acct.get_epsilon(delta=args.delta)
    print(f"Estimated ε = {estimated_epsilon}")

    # # Sets `world_size = 1` if you run on a single GPU with `args.local_rank = -1`
    # if args.local_rank != -1 or args.device != "cpu":
    #     rank, local_rank, world_size = setup(args)
    #     device = local_rank
    # else:
    #     device = "cpu"
    #     rank = 0
    #     world_size = 1

    device = args.device
    rank = 0
    world_size = 1
    torch.set_default_tensor_type(f"torch.{args.device}.FloatTensor")

    if args.secure_rng:
        try:
            import torchcsprng as prng
        except ImportError as e:
            msg = (
                "To use secure RNG, you must install the torchcsprng package! "
                "Check out the instructions here: https://github.com/pytorch/csprng#installation"
            )
            raise ImportError(msg) from e

        generator = prng.create_random_device_generator("/dev/urandom")

    else:
        generator = None

    augmentations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    train_transform = transforms.Compose(
        augmentations + normalize if args.disable_dp else normalize
    )

    test_transform = transforms.Compose(normalize)

    train_dataset = CIFAR10(
        root=args.data_root, train=True, download=True, transform=train_transform
    )

    loader_generator = torch.Generator(device=args.device)
    loader_generator.manual_seed(0)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
    )

    test_dataset = CIFAR10(
        root=args.data_root, train=False, download=True, transform=test_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size_test,
        shuffle=False,
    )

    best_acc = 0

    if args.arch == "simple":
        model = convnet(num_classes=10)
    elif args.arch == "resnet18":
        model = models.resnet18(num_classes=10)
        model = ModuleValidator.fix(model)

    model = model.to(device)

    # # Use the right distributed module wrapper if distributed training is enabled
    # if world_size > 1:
    #     if not args.disable_dp:
    #         if args.clip_per_layer:
    #             model = DDP(model, device_ids=[device])
    #         else:
    #             model = DPDDP(model)
    #     else:
    #         model = DDP(model, device_ids=[device])

    if args.optim == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optim == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optim == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError("Optimizer not recognized. Please check spelling")

    privacy_engine = None
    if not args.disable_dp:
        if args.clip_per_layer:
            # Each layer has the same clipping threshold. The total grad norm is still bounded by `args.max_per_sample_grad_norm`.
            n_layers = len(
                [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            )
            max_grad_norm = [
                args.max_per_sample_grad_norm / np.sqrt(n_layers)
            ] * n_layers
        else:
            max_grad_norm = args.max_per_sample_grad_norm

        privacy_engine = PrivacyEngine(
            secure_mode=args.secure_rng,
        )
        noise_generator = torch.Generator(device=args.device)
        noise_generator.manual_seed(args.seed)
        clipping = "per_layer" if args.clip_per_layer else "flat"
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=args.sigma,
            max_grad_norm=max_grad_norm,
            clipping=clipping,
            grad_sample_mode=args.grad_sample_mode,
            noise_generator=noise_generator,
        )

    # Store some logs
    accuracy_per_epoch = []
    time_per_epoch = []

    # Log run.
    run_name = f"cifar10_{args.arch}_{args.sigma}"
    run_dir = pathlib.Path(args.out_dir) / run_name
    pathlib.Path.mkdir(run_dir, parents=True, exist_ok=True)
    checkpoint_filename = run_dir / f"model_{args.seed}"

    run_params = {
        "project": "multiplicities",
        "job_type": "train",
        "group": run_name,
        "config": {
            "name": run_name,
            "out_path": checkpoint_filename,
            "dataset": "cifar10",
            "model": args.arch,
            "seed": args.seed,
            "sigma": args.sigma,
            "delta": args.delta,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
        },
    }
    run = wandb.init(**run_params, name=f"model_{args.seed}")

    progress = tqdm(range(args.start_epoch, args.epochs + 1))
    for epoch in progress:
        if args.lr_schedule == "cos":
            lr = args.lr * 0.5 * (1 + np.cos(np.pi * epoch / (args.epochs + 1)))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        train_duration, train_loss = train(
            args, model, train_loader, optimizer, privacy_engine, epoch, device
        )
        test_acc, test_loss = test(args, model, test_loader, device)

        if not args.disable_dp:
            epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
            progress.set_postfix_str(
                f"train_loss: {train_loss:.6f} "
                f"test_loss: {test_loss:.6f} "
                f"test_acc@1: {np.mean(test_acc):.6f} "
                f"(ε = {epsilon:.2f}, δ = {args.delta})"
            )
        else:
            progress.set_postfix_str(
                f"train_loss: {train_loss:.6f} "
                f"test_loss: {test_loss:.6f} "
                f"test_acc@1: {np.mean(test_acc):.6f} "
            )

        # remember best acc@1 and save checkpoint
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        time_per_epoch.append(train_duration)
        accuracy_per_epoch.append(float(test_acc))

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "model": args.arch,
                "state_dict": model.state_dict(),
                "test_acc": test_acc,
                "best_acc": best_acc,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            filename=checkpoint_filename,
        )

        if rank == 0:
            time_per_epoch_seconds = [t.total_seconds() for t in time_per_epoch]
            avg_time_per_epoch = sum(time_per_epoch_seconds) / len(
                time_per_epoch_seconds
            )
            if not args.disable_dp:
                epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
            else:
                epsilon = None

            metrics = {
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "avg_time_per_epoch": avg_time_per_epoch,
                "epsilon": epsilon,
            }
            run.log(metrics)
            logger.info(metrics)

    if world_size > 1:
        cleanup()


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 DP Training")
    parser.add_argument("--grad_sample_mode", type=str, default="hooks")
    parser.add_argument(
        "-j",
        "--workers",
        default=1,
        type=int,
        metavar="N",
        help="number of data loading workers",
    )
    parser.add_argument(
        "--arch", type=str, choices=["simple", "resnet18"], default="simple"
    )
    parser.add_argument(
        "--epochs",
        default=30,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch",
        default=1,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size-test",
        default=128,
        type=int,
        metavar="N",
        help="mini-batch size for test dataset (default: 256), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--batch-size",
        default=64,
        type=int,
        metavar="N",
        help="approximate bacth size",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.001,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="SGD momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=0,
        type=float,
        metavar="W",
        help="SGD weight decay",
        dest="weight_decay",
    )

    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )

    parser.add_argument(
        "--sigma",
        default=None,
        metavar="S",
        help="Noise multiplier",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=10.0,
        metavar="C",
        help="Clip per-sample gradients to this norm (default 1.0)",
    )
    parser.add_argument(
        "--secure-rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees."
        "Comes at a performance cost. Opacus will emit a warning if secure rng is off,"
        "indicating that for production use it's recommender to turn it on.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta (default: 1e-5)",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="out",
        help="path to save check points",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../cifar10",
        help="Where CIFAR10 is/will be stored",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="/tmp/stat/tensorboard",
        help="Where Tensorboard log will be stored",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="Adam",
        help="Optimizer to use (Adam, RMSprop, SGD)",
    )
    parser.add_argument(
        "--lr-schedule", type=str, choices=["constant", "cos"], default="constant"
    )

    parser.add_argument(
        "--device", type=str, default="cuda", help="Device on which to run the code."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank if multi-GPU training, -1 for single GPU training. Will be overriden by the environment variables if running on a Slurm cluster.",
    )

    parser.add_argument(
        "--dist_backend",
        type=str,
        default="gloo",
        help="Choose the backend for torch distributed from: gloo, nccl, mpi",
    )

    parser.add_argument(
        "--clip_per_layer",
        action="store_true",
        default=False,
        help="Use static per-layer clipping with the same clipping threshold for each layer. Necessary for DDP. If `False` (default), uses flat clipping.",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        help="debug level (default: 0)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
