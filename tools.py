import argparse
import os
from src.helpers import download_dataset, verify_dataset_structure, generate_analytics_plots


def main():
    parser = argparse.ArgumentParser(description="Segmenti Utility Processing CLI")
    parser.add_argument(
        "--mode", type=str, required=True, choices=["download", "verify", "plot"], help="Utility phase option"
    )
    parser.add_argument("--root_dir", type=str, default="dataset", help="Dataset target folder root")
    parser.add_argument("--dataset_name", type=str, default="plantsegv3", help="Target folder alignment name")

    args = parser.parse_args()

    if args.mode == "download":
        print("Constructing default workspace directories...")
        download_dataset(target_dir=args.root_dir)

    elif args.mode == "verify":
        target_path = os.path.join(args.root_dir, args.dataset_name)
        verify_dataset_structure(root_dir=target_path)

    elif args.mode == "plot":
        print("Processing training history diagnostics plots...")
        generate_analytics_plots()


if __name__ == "__main__":
    main()
