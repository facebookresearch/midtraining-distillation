# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from lingua.checkpoint import consolidate_checkpoints
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    args = parser.parse_args()
    consolidate_path = consolidate_checkpoints(args.ckpt_dir)
    print(f"Consolidated checkpoint saved to {consolidate_path}")


if __name__ == "__main__":
    main()
