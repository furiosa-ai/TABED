#!/usr/bin/env python3
"""Dataset download utilities for TABED.

This script downloads and prepares various datasets for vision-language
model evaluation. Run with --help for usage information.

Usage:
    python -m tabed.datamodules.download_datasets --save-dir /path/to/data
    python -m tabed.datamodules.download_datasets --dataset vibe-eval
    python -m tabed.datamodules.download_datasets --all
"""

import argparse
import os
from typing import List, Optional

from datasets import DatasetDict, load_dataset


# =============================================================================
# Dataset Configuration
# =============================================================================

# Single-image evaluation datasets
SINGLE_IMAGE_REPOS = [
    "lmms-lab/DC100_EN",
    "lmms-lab/llava-bench-in-the-wild",
    "RekaAI/VibeEval",
    "lmms-lab/LiveBench",
    "lmms-lab/MMVet",
    "lmms-lab/POPE",
    "lmms-lab/HallusionBench",
]

# Multi-image interleave datasets (from LLaVA-NeXT-Interleave-Bench)
MULTI_IMAGE_DATASETS = [
    'Spot-the-Diff',
    'Birds-to-Words',
    'CLEVR-Change',
    'HQ-Edit',
    'MagicBrush',
    'IEdit',
    'AESOP',
    'FlintstonesSV',
    'PororoSV',
    'VIST',
    'WebQA',
    'QBench',
    'NLVR2_Mantis',
    'OCR-VQA',
]

# LMMs-Eval-Lite subsets
LMMS_EVAL_LITE_SUBSETS = [
    'chartqa',
    'docvqa_val',
    'infovqa_val',
    'ok_vqa_val2014',
    'textvqa_val',
    'vizwiz_vqa_val',
    'vqav2_val',
]

# Other datasets
OTHER_REPOS = [
    "google-research-datasets/nq_open",
]

# Dataset name to category mapping
DATASET_CATEGORIES = {
    'single': SINGLE_IMAGE_REPOS,
    'multi': MULTI_IMAGE_DATASETS,
    'lmms-eval': LMMS_EVAL_LITE_SUBSETS,
    'other': OTHER_REPOS,
}


# =============================================================================
# Download Functions
# =============================================================================

def download_single_image_datasets(
    save_dir: str,
    repos: Optional[List[str]] = None,
) -> None:
    """Download single-image evaluation datasets.

    Args:
        save_dir: Directory to save datasets.
        repos: Optional list of specific repos to download. Downloads all if None.
    """
    repos = repos or SINGLE_IMAGE_REPOS

    for repo in repos:
        print(f"\n[Downloading] {repo}")

        try:
            if repo == "lmms-lab/LiveBench":
                datasets = load_dataset(repo, '2024-05')
            else:
                datasets = load_dataset(repo)

            dataset_name = repo.split("/")[-1]
            save_path = os.path.join(save_dir, dataset_name)

            # Print features for first split
            for split, dataset in datasets.items():
                print(f"  Features: {list(dataset.features.keys())}")
                print(f"  Size: {len(dataset)} samples")
                break

            # Handle special cases for split naming
            if repo == "lmms-lab/llava-bench-in-the-wild":
                datasets = DatasetDict({'test': datasets['train']})
            elif repo == "lmms-lab/HallusionBench":
                datasets = DatasetDict({'test': datasets['image']})

            datasets.save_to_disk(save_path)
            print(f"  Saved to: {save_path}")

        except Exception as e:
            print(f"  Error: {e}")


def download_multi_image_datasets(
    save_dir: str,
    dataset_names: Optional[List[str]] = None,
) -> None:
    """Download multi-image interleave datasets.

    Args:
        save_dir: Directory to save datasets.
        dataset_names: Optional list of specific datasets. Downloads all if None.
    """
    dataset_names = dataset_names or MULTI_IMAGE_DATASETS

    print(f"\n[Downloading] lmms-lab/LLaVA-NeXT-Interleave-Bench")

    try:
        repo = "lmms-lab/LLaVA-NeXT-Interleave-Bench"
        datasets = load_dataset(repo, "in_domain")

        # Print features
        for split, dataset in datasets.items():
            print(f"  Features: {list(dataset.features.keys())}")
            break

        # Filter and save each dataset
        datasets_filtered = datasets.filter(
            lambda x: x['sub_task'] in dataset_names
        )

        for dataset_name in dataset_names:
            save_path = os.path.join(save_dir, dataset_name)
            sub_datasets = datasets_filtered.filter(
                lambda x: x['sub_task'] == dataset_name
            )

            if len(sub_datasets['test']) > 0:
                sub_datasets.save_to_disk(save_path)
                print(f"  Saved {dataset_name}: {len(sub_datasets['test'])} samples")
            else:
                print(f"  Skipped {dataset_name}: no samples found")

    except Exception as e:
        print(f"  Error: {e}")


def download_lmms_eval_lite(
    save_dir: str,
    subsets: Optional[List[str]] = None,
) -> None:
    """Download LMMs-Eval-Lite datasets.

    Args:
        save_dir: Directory to save datasets.
        subsets: Optional list of specific subsets. Downloads all if None.
    """
    subsets = subsets or LMMS_EVAL_LITE_SUBSETS
    repo = "lmms-lab/LMMs-Eval-Lite"

    print(f"\n[Downloading] {repo}")

    for subset in subsets:
        print(f"  Subset: {subset}")

        try:
            datasets = load_dataset(repo, subset)
            save_path = os.path.join(save_dir, subset)

            # Print features
            for split, dataset in datasets.items():
                print(f"    Features: {list(dataset.features.keys())}")
                print(f"    Size: {len(dataset)} samples")
                break

            # Convert to test split
            datasets = DatasetDict({'test': datasets['lite']})
            datasets.save_to_disk(save_path)
            print(f"    Saved to: {save_path}")

        except Exception as e:
            print(f"    Error: {e}")


def download_other_datasets(
    save_dir: str,
    repos: Optional[List[str]] = None,
) -> None:
    """Download other datasets (NQ, etc.).

    Args:
        save_dir: Directory to save datasets.
        repos: Optional list of specific repos. Downloads all if None.
    """
    repos = repos or OTHER_REPOS

    for repo in repos:
        print(f"\n[Downloading] {repo}")

        try:
            datasets = load_dataset(repo)
            dataset_name = repo.split("/")[-1]
            save_path = os.path.join(save_dir, dataset_name)

            # Print features
            for split, dataset in datasets.items():
                print(f"  Features: {list(dataset.features.keys())}")
                print(f"  Size: {len(dataset)} samples")
                break

            datasets.save_to_disk(save_path)
            print(f"  Saved to: {save_path}")

        except Exception as e:
            print(f"  Error: {e}")


def download_all(save_dir: str) -> None:
    """Download all available datasets.

    Args:
        save_dir: Directory to save datasets.
    """
    print("=" * 60)
    print("Downloading all datasets")
    print("=" * 60)

    download_single_image_datasets(save_dir)
    download_multi_image_datasets(save_dir)
    download_lmms_eval_lite(save_dir)
    download_other_datasets(save_dir)

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)


# =============================================================================
# CLI Interface
# =============================================================================

def get_default_save_dir() -> str:
    """Get the default save directory from environment or fallback.

    Returns:
        Default save directory path.
    """
    return os.environ.get(
        'TABED_DATA_DIR',
        os.path.join(os.path.expanduser('~'), 'data', 'tabed', 'datasets')
    )


def main():
    """Main entry point for dataset download CLI."""
    parser = argparse.ArgumentParser(
        description='Download datasets for TABED evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download all datasets to default directory
    python -m tabed.datamodules.download_datasets --all

    # Download to specific directory
    python -m tabed.datamodules.download_datasets --all --save-dir /data/tabed

    # Download specific category
    python -m tabed.datamodules.download_datasets --category single

    # Download specific dataset by name
    python -m tabed.datamodules.download_datasets --dataset VibeEval

Environment Variables:
    TABED_DATA_DIR: Default save directory for datasets
        """
    )

    parser.add_argument(
        '--save-dir',
        type=str,
        default=None,
        help='Directory to save datasets (default: ~/data/tabed/datasets or $TABED_DATA_DIR)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all available datasets'
    )
    parser.add_argument(
        '--category',
        type=str,
        choices=['single', 'multi', 'lmms-eval', 'other'],
        help='Download datasets from a specific category'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Download a specific dataset by name'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available datasets'
    )

    args = parser.parse_args()

    # Handle --list
    if args.list:
        print("\nAvailable datasets:")
        print("\nSingle-image datasets:")
        for repo in SINGLE_IMAGE_REPOS:
            print(f"  - {repo.split('/')[-1]}")
        print("\nMulti-image datasets:")
        for name in MULTI_IMAGE_DATASETS:
            print(f"  - {name}")
        print("\nLMMs-Eval-Lite subsets:")
        for name in LMMS_EVAL_LITE_SUBSETS:
            print(f"  - {name}")
        print("\nOther datasets:")
        for repo in OTHER_REPOS:
            print(f"  - {repo.split('/')[-1]}")
        return

    # Determine save directory
    save_dir = args.save_dir or get_default_save_dir()
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory: {save_dir}")

    # Download based on arguments
    if args.all:
        download_all(save_dir)
    elif args.category:
        if args.category == 'single':
            download_single_image_datasets(save_dir)
        elif args.category == 'multi':
            download_multi_image_datasets(save_dir)
        elif args.category == 'lmms-eval':
            download_lmms_eval_lite(save_dir)
        elif args.category == 'other':
            download_other_datasets(save_dir)
    elif args.dataset:
        # Find and download specific dataset
        dataset_name = args.dataset

        # Check single-image repos
        matching_repos = [r for r in SINGLE_IMAGE_REPOS if dataset_name in r]
        if matching_repos:
            download_single_image_datasets(save_dir, matching_repos)
            return

        # Check multi-image datasets
        if dataset_name in MULTI_IMAGE_DATASETS:
            download_multi_image_datasets(save_dir, [dataset_name])
            return

        # Check LMMs-Eval-Lite
        if dataset_name in LMMS_EVAL_LITE_SUBSETS:
            download_lmms_eval_lite(save_dir, [dataset_name])
            return

        # Check other repos
        matching_repos = [r for r in OTHER_REPOS if dataset_name in r]
        if matching_repos:
            download_other_datasets(save_dir, matching_repos)
            return

        print(f"Dataset '{dataset_name}' not found. Use --list to see available datasets.")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
