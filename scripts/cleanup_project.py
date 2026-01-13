from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def _repo_root() -> Path:
    start = Path(__file__).resolve().parent
    for candidate in [start, *start.parents]:
        if (
            (candidate / "services").is_dir()
            and (candidate / "train").is_dir()
            and (candidate / "eval").is_dir()
            and (candidate / "mock_apis").is_dir()
        ):
            return candidate
    raise RuntimeError(f"Could not determine repo root from: {start}")


def _resolve_under_repo(p: Path, repo_root: Path) -> Path:
    resolved = p.resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as e:
        raise ValueError(f"Refusing to touch path outside repo: {resolved}") from e
    return resolved


def _iter_eval_result_dirs(results_dir: Path) -> List[Path]:
    if not results_dir.exists():
        return []
    out: List[Path] = []
    for child in results_dir.iterdir():
        if child.is_dir() and re.fullmatch(r"\d{8}T\d{6}Z", child.name):
            out.append(child)
    return sorted(out)


def _iter_train_model_files(models_dir: Path) -> List[Path]:
    if not models_dir.exists():
        return []
    out: List[Path] = []
    for child in models_dir.iterdir():
        if not child.is_file():
            continue
        if child.name == "__init__.py":
            continue
        if child.suffix in {".pkl", ".json"} and (
            child.name.endswith(".metrics.json") or child.suffix == ".pkl"
        ):
            out.append(child)
    return sorted(out)


def _iter_mock_s3_training_objects(s3_training_dir: Path) -> List[Path]:
    if not s3_training_dir.exists():
        return []
    out: List[Path] = []
    for child in s3_training_dir.iterdir():
        if child.is_file():
            out.append(child)
    return sorted(out)


def _iter_rm_under_dir(
    target_dir: Path, *, keep_file_names: Optional[set[str]] = None
) -> List[Tuple[str, Path, Optional[bytes]]]:
    if not target_dir.exists():
        return []
    keep_file_names = keep_file_names or set()

    actions: List[Tuple[str, Path, Optional[bytes]]] = []
    for child in target_dir.iterdir():
        if child.is_file():
            if child.name in keep_file_names:
                continue
            actions.append(("rm_file", child, None))
            continue
        if child.is_dir():
            actions.append(("rm_dir", child, None))
            continue
    return sorted(actions, key=lambda a: str(a[1]))


def _plan_actions(repo_root: Path) -> List[Tuple[str, Path, Optional[bytes]]]:
    """Return a list of actions.

    Each action is a tuple:
      (kind, path, content)

    - kind == "rm_file": delete file
    - kind == "rm_dir": delete directory
    - kind == "write_bytes": write content (overwrites)
    """

    actions: List[Tuple[str, Path, Optional[bytes]]] = []

    eval_results_dir = repo_root / "eval" / "model_evaluation" / "results"
    actions.extend(_iter_rm_under_dir(eval_results_dir))

    eval_data_dir = repo_root / "eval" / "model_evaluation" / "data"
    actions.extend(_iter_rm_under_dir(eval_data_dir))

    eval_reports_dir = repo_root / "eval" / "reports"
    actions.extend(_iter_rm_under_dir(eval_reports_dir))

    train_data_dir = repo_root / "train" / "data"
    actions.extend(_iter_rm_under_dir(train_data_dir))

    train_models_dir = repo_root / "train" / "models"
    actions.extend(_iter_rm_under_dir(train_models_dir, keep_file_names={"__init__.py"}))

    mock_s3_training_dir = repo_root / "mock_apis" / "cloud_apis" / "data" / "s3" / "training"
    actions.extend(_iter_rm_under_dir(mock_s3_training_dir))

    dynamodb_dir = repo_root / "mock_apis" / "cloud_apis" / "data" / "dynamodb"
    actions.extend(_iter_rm_under_dir(dynamodb_dir))
    actions.append(("write_bytes", dynamodb_dir / "training_runs.json", b"[]\n"))
    actions.append(("write_bytes", dynamodb_dir / "orchestrator_audit.json", b"[]\n"))

    cloudwatch_dir = repo_root / "mock_apis" / "cloud_apis" / "data" / "cloudwatch"
    actions.extend(_iter_rm_under_dir(cloudwatch_dir))
    actions.append(("write_bytes", cloudwatch_dir / "metrics.jsonl", b""))

    return actions


def _maybe_add_registry_reset(
    *,
    actions: List[Tuple[str, Path, Optional[bytes]]],
    repo_root: Path,
    reset_current_model: bool,
) -> None:
    if not reset_current_model:
        return

    registry_dir = repo_root / "mock_apis" / "cloud_apis" / "data" / "registry"
    current_model_path = registry_dir / "current_model.json"

    # A minimal file that makes mock_model_registry.get_current_model(task=...) return None.
    data = {"tasks": {}}
    actions.append(("write_bytes", current_model_path, (json.dumps(data, indent=2) + "\n").encode("utf-8")))


def _apply_actions(
    actions: Iterable[Tuple[str, Path, Optional[bytes]]],
    *,
    repo_root: Path,
    dry_run: bool,
) -> None:
    for kind, path, content in actions:
        resolved = _resolve_under_repo(path, repo_root)

        if kind == "rm_dir":
            if dry_run:
                print(f"[dry-run] rm -r {resolved}")
                continue
            if resolved.exists():
                shutil.rmtree(resolved)
            continue

        if kind == "rm_file":
            if dry_run:
                print(f"[dry-run] rm {resolved}")
                continue
            if resolved.exists():
                resolved.unlink()
            continue

        if kind == "write_bytes":
            if content is None:
                raise ValueError(f"write_bytes requires content: {resolved}")
            if dry_run:
                print(f"[dry-run] write {resolved} ({len(content)} bytes)")
                continue
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_bytes(content)
            continue

        raise ValueError(f"Unknown action kind: {kind}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Cleanup script to remove local run artifacts and reset mock cloud state. "
            "Runs in dry-run mode unless you pass --yes."
        )
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually perform deletions/rewrites (otherwise dry-run).",
    )
    parser.add_argument(
        "--reset-registry-current-model",
        action="store_true",
        help="Reset mock registry current_model.json to an empty tasks mapping.",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    actions = _plan_actions(repo_root)
    _maybe_add_registry_reset(
        actions=actions,
        repo_root=repo_root,
        reset_current_model=bool(args.reset_registry_current_model),
    )

    dry_run = not bool(args.yes)
    if dry_run:
        print("Dry run (no changes will be made). Pass --yes to apply.")

    _apply_actions(actions, repo_root=repo_root, dry_run=dry_run)

    if dry_run:
        print("Done (dry-run).")
    else:
        print("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
