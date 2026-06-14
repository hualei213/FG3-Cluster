#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build weakly-supervised ISIC 2018 metadata splits.

This script:
1. Reads the official ISIC Archive Challenge 2018 Task 1-2 metadata CSV files:
   - training
   - validation
   - test
2. Unifies their columns.
3. Maps diagnosis_2 into Scheme-A labels:
   A1_Benign_melanocytic
   A2_Melanoma
   A3_Benign_epidermal
   A4_Other
4. Sorts rows by diagnosis_A group.
5. Creates a stratified 7:3 train/test split by diagnosis_A.
6. Saves two final CSV files with exactly two columns:
   isic_id, diagnosis_A

Example:
    python split_isic2018_schemeA.py \
        --input-dir /path/to/isic_metadata \
        --output-dir /path/to/output \
        --seed 42 \
        --train-ratio 0.7

If your filenames are different, pass them explicitly:
    python split_isic2018_schemeA.py \
        --train-metadata challenge-2018-task-1-2-training_metadata_XXXX-XX-XX.csv \
        --val-metadata challenge-2018-task-1-2-validation_metadata_XXXX-XX-XX.csv \
        --test-metadata challenge-2018-task-1-2-test_metadata_XXXX-XX-XX.csv \
        --output-dir ./outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


SCHEME_A_MAP: Dict[str, str] = {
    "Benign melanocytic proliferations": "A1_Benign_melanocytic",
    "Malignant melanocytic proliferations (Melanoma)": "A2_Melanoma",
    "Benign epidermal proliferations": "A3_Benign_epidermal",
}

SCHEME_A_ORDER: List[str] = [
    "A1_Benign_melanocytic",
    "A2_Melanoma",
    "A3_Benign_epidermal",
    "A4_Other",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge ISIC 2018 Task 1-2 metadata and create Scheme-A stratified 7:3 splits."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("../.."),
        help="Directory containing the three metadata CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./isic2018_schemeA_split"),
        help="Directory to save generated CSV files.",
    )

    parser.add_argument(
        "--train-metadata",
        type=Path,
        default=None,
        help="Path to training metadata CSV. If omitted, default filename under --input-dir is used.",
    )
    parser.add_argument(
        "--val-metadata",
        type=Path,
        default=None,
        help="Path to validation metadata CSV. If omitted, default filename under --input-dir is used.",
    )
    parser.add_argument(
        "--test-metadata",
        type=Path,
        default=None,
        help="Path to test metadata CSV. If omitted, default filename under --input-dir is used.",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train ratio for stratified split. Default: 0.7.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible stratified split. Default: 42.",
    )
    parser.add_argument(
        "--allow-duplicate-isic-id",
        action="store_true",
        help="Allow duplicated isic_id values. By default, duplicated isic_id will raise an error.",
    )

    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    default_train = args.input_dir / "challenge-2018-task-1-2-training_metadata_2025-11-12.csv"
    default_val = args.input_dir / "challenge-2018-task-1-2-validation_metadata_2025-12-26.csv"
    default_test = args.input_dir / "challenge-2018-task-1-2-test_metadata_2025-12-26.csv"

    train_path = args.train_metadata if args.train_metadata is not None else default_train
    val_path = args.val_metadata if args.val_metadata is not None else default_val
    test_path = args.test_metadata if args.test_metadata is not None else default_test

    train_path = train_path.expanduser().resolve()
    val_path = val_path.expanduser().resolve()
    test_path = test_path.expanduser().resolve()

    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")

    return train_path, val_path, test_path


def read_metadata(path: Path, split_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "isic_id" not in df.columns:
        raise ValueError(f"{path} does not contain required column: isic_id")
    if "diagnosis_2" not in df.columns:
        raise ValueError(f"{path} does not contain required column: diagnosis_2")
    df = df.copy()
    df["official_split"] = split_name
    return df


def build_unified_dataframe(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    # Use validation/test header first because these files usually contain mel_thick_mm and mel_ulcer.
    # Then append any remaining columns from training to avoid information loss.
    preferred_cols: List[str] = []
    for cols in [list(val_df.columns), list(test_df.columns), list(train_df.columns)]:
        for col in cols:
            if col not in preferred_cols:
                preferred_cols.append(col)

    train_df = train_df.reindex(columns=preferred_cols)
    val_df = val_df.reindex(columns=preferred_cols)
    test_df = test_df.reindex(columns=preferred_cols)

    merged = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return merged


def add_scheme_a_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    missing_diag = out["diagnosis_2"].isna().sum()
    if missing_diag > 0:
        raise ValueError(f"Found {missing_diag} rows with missing diagnosis_2. Please handle them before splitting.")

    out["diagnosis_A"] = out["diagnosis_2"].map(SCHEME_A_MAP).fillna("A4_Other")
    return out


def sort_by_scheme_a(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rank = {name: i for i, name in enumerate(SCHEME_A_ORDER)}
    out["_scheme_a_rank"] = out["diagnosis_A"].map(rank).fillna(999).astype(int)
    out["_orig_idx"] = np.arange(len(out))

    out = out.sort_values(["_scheme_a_rank", "_orig_idx"], kind="mergesort")
    out = out.drop(columns=["_scheme_a_rank", "_orig_idx"]).reset_index(drop=True)
    return out


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"--train-ratio must be between 0 and 1, got {train_ratio}")

    rng = np.random.default_rng(seed)
    split_df = df[["isic_id", "diagnosis_A"]].copy()

    train_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []

    for class_name in SCHEME_A_ORDER:
        group = split_df[split_df["diagnosis_A"] == class_name]
        if group.empty:
            continue

        idx = group.index.to_numpy()
        rng.shuffle(idx)

        n = len(idx)
        n_train = int(round(n * train_ratio))

        # Keep both train and test non-empty when the class has at least 2 samples.
        if n >= 2:
            n_train = max(1, min(n - 1, n_train))
        else:
            n_train = n

        train_idx = idx[:n_train]
        test_idx = idx[n_train:]

        train_parts.append(split_df.loc[train_idx])
        test_parts.append(split_df.loc[test_idx])

    train_out = pd.concat(train_parts, ignore_index=True)
    test_out = pd.concat(test_parts, ignore_index=True)

    # Shuffle each final split while preserving reproducibility.
    train_out = train_out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_out = test_out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    summary = make_summary(split_df, train_out, test_out)
    return train_out, test_out, summary


def make_summary(all_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    all_counts = all_df["diagnosis_A"].value_counts().reindex(SCHEME_A_ORDER, fill_value=0)
    train_counts = train_df["diagnosis_A"].value_counts().reindex(SCHEME_A_ORDER, fill_value=0)
    test_counts = test_df["diagnosis_A"].value_counts().reindex(SCHEME_A_ORDER, fill_value=0)

    summary = pd.DataFrame(
        {
            "diagnosis_A": SCHEME_A_ORDER,
            "all": all_counts.values,
            "train": train_counts.values,
            "test": test_counts.values,
        }
    )
    summary["train_ratio"] = (summary["train"] / summary["all"]).replace([np.inf, -np.inf], np.nan).round(4)
    summary["test_ratio"] = (summary["test"] / summary["all"]).replace([np.inf, -np.inf], np.nan).round(4)
    return summary


def validate_outputs(
    merged_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    allow_duplicate_isic_id: bool = False,
) -> None:
    required_cols = ["isic_id", "diagnosis_A"]
    if list(train_df.columns) != required_cols:
        raise AssertionError(f"Train CSV columns should be {required_cols}, got {list(train_df.columns)}")
    if list(test_df.columns) != required_cols:
        raise AssertionError(f"Test CSV columns should be {required_cols}, got {list(test_df.columns)}")

    if not allow_duplicate_isic_id:
        dup_count = merged_df["isic_id"].duplicated().sum()
        if dup_count > 0:
            dup_ids = merged_df.loc[merged_df["isic_id"].duplicated(), "isic_id"].head(10).tolist()
            raise ValueError(f"Found {dup_count} duplicated isic_id values, e.g. {dup_ids}")

    overlap = set(train_df["isic_id"]).intersection(set(test_df["isic_id"]))
    if overlap:
        example = sorted(list(overlap))[:10]
        raise AssertionError(f"Train/test overlap detected: {example}")


def main() -> None:
    args = parse_args()
    train_path, val_path, test_path = resolve_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = read_metadata(train_path, "official_train")
    val_df = read_metadata(val_path, "official_val")
    test_df = read_metadata(test_path, "official_test")

    merged_df = build_unified_dataframe(train_df, val_df, test_df)
    labeled_df = add_scheme_a_label(merged_df)
    grouped_df = sort_by_scheme_a(labeled_df)

    train_split, test_split, summary = stratified_split(
        grouped_df,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    validate_outputs(
        merged_df=grouped_df,
        train_df=train_split,
        test_df=test_split,
        allow_duplicate_isic_id=args.allow_duplicate_isic_id,
    )

    merged_path = args.output_dir / "challenge-2018-task-1-2-all_metadata_merged.csv"
    grouped_path = args.output_dir / "challenge-2018-task-1-2-metadata_grouped_by_schemeA.csv"
    train_out_path = args.output_dir / "isic2018_schemeA_train_7_3.csv"
    test_out_path = args.output_dir / "isic2018_schemeA_test_7_3.csv"
    summary_path = args.output_dir / "isic2018_schemeA_split_summary.csv"

    merged_df.to_csv(merged_path, index=False)
    grouped_df.to_csv(grouped_path, index=False)
    train_split.to_csv(train_out_path, index=False)
    test_split.to_csv(test_out_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Done.")
    print(f"Merged metadata: {merged_path}")
    print(f"Grouped metadata: {grouped_path}")
    print(f"Train split:     {train_out_path}")
    print(f"Test split:      {test_out_path}")
    print(f"Summary:         {summary_path}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
