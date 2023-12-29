#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import logging
import os
import crypten

from crypten.examples.multiprocess_launcher import MultiProcessLauncher

parser = argparse.ArgumentParser(description="CrypTen Bladder Iraining")


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
parser.add_argument('--channels', type=int, default=3, help='image channels', dest='channels')
parser.add_argument('--classes', type=int, default=1, help='mask nums', dest='classes')
parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
parser.add_argument('-v', '--validation', dest='val', type=float, default=20.0,
                        help='Percent of the data that is used as validation (0-100)')


def _run_experiment(args):
    from test import run_inference

    run_inference(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        momentum=args.momentum,
        channels=args.channels,
        classes=args.classes,
        scale=args.scale,
        val_percent=args.val/100
    )

def main(run_experiment):
    args = parser.parse_args()
    launcher = MultiProcessLauncher(args.world_size, run_experiment, args)
    launcher.start()
    launcher.join()
    launcher.terminate()

if __name__ == "__main__":
    main(_run_experiment)
