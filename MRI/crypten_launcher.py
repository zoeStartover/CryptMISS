#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
from crypten.examples.multiprocess_launcher import MultiProcessLauncher

parser = argparse.ArgumentParser(description="CrypTen LGG Iraining")


def validate_world_size(world_size):
    world_size = int(world_size)
    if world_size < 2:
        raise argparse.ArgumentTypeError(f"world_size {world_size} must be > 1")
    return world_size


parser.add_argument(
    "--world_size",
    type=validate_world_size,
    default=2,
    help="The number of parties to launch. Each party acts as its own process",
)
parser.add_argument(
    "--epochs", default=15, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.0001,
    type=float,
    metavar="LR",
    help="initial learning rate",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=1,
    type=int,
    metavar="N",
    help="mini-batch size (default: 5)",
)
parser.add_argument(
    "--print-freq",
    "-p",
    default=5,
    type=int,
    metavar="PF",
    help="print frequency (default: 5)",
)
parser.add_argument(
    "--num-samples",
    "-n",
    default=2000,
    type=int,
    metavar="N",
    help="num of samples used for training (default: 100)",
)


def _run_experiment(args):
    from crypten_inference import run_lgg_inference

    run_lgg_inference(
        batch_size=args.batch_size,
    )

    # run_lgg_train(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, print_freq=args.print_freq)

def main(run_experiment):
    args = parser.parse_args()
    launcher = MultiProcessLauncher(args.world_size, run_experiment, args)
    launcher.start()
    launcher.join()
    launcher.terminate()

if __name__ == "__main__":
    main(_run_experiment)
