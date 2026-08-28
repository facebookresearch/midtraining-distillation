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
