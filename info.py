#!/usr/bin/env python3
"""Compare per-class image counts between two dataset folders."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    default_root = project_root / "polyu_vegetation" / "extracted_grouped"
    default_target = project_root / "polyu_vegetation" / "extracted_group_augmented"
    parser = argparse.ArgumentParser(
        description=(
            "Compare per-class image counts between root and target datasets. "
            "Paths can be absolute, or relative to this script directory."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="Source dataset directory (default: polyu_vegetation/extracted_grouped).",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=default_target,
        help="Target dataset directory to compare against root.",
    )
    return parser.parse_args()


def count_files_with_suffix(folder: Path, suffixes: set[str]) -> int:
    if not folder.exists() or not folder.is_dir():
        return 0
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def resolve_path(p: Path) -> Path:
    if p.is_absolute():
        return p.resolve()
    return (Path(__file__).resolve().parent / p).resolve()


def main() -> None:
    args = parse_args()

    root_dir = resolve_path(args.root)
    target_dir = resolve_path(args.target)

    if not root_dir.exists() or not root_dir.is_dir():
        raise FileNotFoundError(f"--root dataset not found: {root_dir}")
    if not target_dir.exists() or not target_dir.is_dir():
        raise FileNotFoundError(f"--target dataset not found: {target_dir}")

    root_classes = {d.name for d in root_dir.iterdir() if d.is_dir()}
    target_classes = {d.name for d in target_dir.iterdir() if d.is_dir()}
    classes = sorted(root_classes | target_classes)

    if not classes:
        print("No class subdirectories found.")
        return

    rows: list[tuple[str, int, int, int, float]] = []
    total_root = 0
    total_target = 0

    for class_name in classes:
        root_count = count_files_with_suffix(root_dir / class_name, {".jpeg", ".jpg", ".png", ".bmp", ".webp"})
        target_count = count_files_with_suffix(
            target_dir / class_name, {".jpeg", ".jpg", ".png", ".bmp", ".webp"}
        )
        delta = target_count - root_count
        fold = (target_count / root_count) if root_count > 0 else 0.0

        rows.append((class_name, root_count, target_count, delta, fold))
        total_root += root_count
        total_target += target_count

    total_delta = total_target - total_root
    total_fold = (total_target / total_root) if total_root > 0 else 0.0

    name_width = max(len("class"), max(len(name) for name, _, _, _, _ in rows))
    root_width = max(len("root"), len(str(total_root)))
    target_width = max(len("target"), len(str(total_target)))
    delta_width = max(len("delta"), len(str(total_delta)))
    fold_width = len("fold")

    header = (
        f"{'class':<{name_width}}  {'root':>{root_width}}  "
        f"{'target':>{target_width}}  {'delta':>{delta_width}}  {'fold':>{fold_width}}"
    )
    divider = "-" * len(header)

    print(f"ROOT  : {root_dir}")
    print(f"TARGET: {target_dir}")
    print(header)
    print(divider)
    for class_name, root_count, target_count, delta, fold in rows:
        print(
            f"{class_name:<{name_width}}  {root_count:>{root_width}}  "
            f"{target_count:>{target_width}}  {delta:>{delta_width}}  {fold:>{fold_width}.2f}x"
        )
    print(divider)
    print(
        f"{'TOTAL':<{name_width}}  {total_root:>{root_width}}  "
        f"{total_target:>{target_width}}  {total_delta:>{delta_width}}  {total_fold:>{fold_width}.2f}x"
    )


if __name__ == "__main__":
    main()
