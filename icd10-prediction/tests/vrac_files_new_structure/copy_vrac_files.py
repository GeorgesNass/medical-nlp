import os
import shutil
import stat
from pathlib import Path


# =========================
# Configuration
# =========================
input_folder = r"."
vrac_files_folder = r"./vrac_files_new_structure"

# Exclusions
excluded_dir_names = {"tmp", "__pycache__", ".git", ".ocr_env", ".pytest_cache", "data", ".monitor_env", ".venv", ".rag_env", ".lq_env", ".lf_env", ".lab_env", ".icd10_env", ".llm_env", ".llm_platf", ".dedu_env", ".dc_env"}  # <= NOMS de dossiers (pas "./tmp")
excluded_exts = {".log", ".zip", "txt", ".pdf", ".TXT", ".PDF"}  # extensions à ignorer (case-insensitive)


def ensure_writable(p: Path) -> None:
    """On Windows, remove read-only attribute if needed."""
    try:
        p.chmod(p.stat().st_mode | stat.S_IWRITE)
    except Exception:
        pass


def is_in_excluded_tree(path: Path, excluded_names: set[str]) -> bool:
    """True if the path is inside a directory whose name is excluded (at any level)."""
    return any(part in excluded_names for part in path.parts)


def copy_all_files_flatten(
    input_root: Path,
    output_root: Path,
    excluded_names: set[str],
    excluded_extensions: set[str],
    overwrite: bool = True,
) -> None:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    errors = 0

    for current_dir, dirnames, filenames in os.walk(input_root):
        current_path = Path(current_dir)

        # Skip output folder (avoid re-copying what we just copied)
        if output_root == current_path or output_root in current_path.parents:
            dirnames[:] = []
            continue

        # Skip excluded trees
        if is_in_excluded_tree(current_path, excluded_names):
            dirnames[:] = []
            skipped += len(filenames)
            continue

        # Prevent descending into excluded directories by name
        dirnames[:] = [d for d in dirnames if d not in excluded_names]

        for fname in filenames:
            src = current_path / fname
            if not src.is_file():
                continue

            # Exclude extensions (case-insensitive)
            if src.suffix.lower() in {e.lower() for e in excluded_extensions}:
                skipped += 1
                continue

            dst = output_root / src.name

            try:
                if dst.exists():
                    if dst.is_dir():
                        # collision: dst is a directory but we want a file
                        if overwrite:
                            shutil.rmtree(dst)
                        else:
                            skipped += 1
                            continue
                    else:
                        if not overwrite:
                            skipped += 1
                            continue
                        ensure_writable(dst)

                shutil.copy2(src, dst)
                copied += 1

            except Exception as e:
                errors += 1
                print(f"[ERROR] {src} -> {dst} | {type(e).__name__}: {e}")

    print("========== DONE ==========")
    print(f"Copied files : {copied}")
    print(f"Skipped files: {skipped}")
    print(f"Errors       : {errors}")
    print(f"Destination  : {output_root}")


if __name__ == "__main__":
    copy_all_files_flatten(
        input_root=Path(input_folder),
        output_root=Path(vrac_files_folder),
        excluded_names=excluded_dir_names,
        excluded_extensions=excluded_exts,
        overwrite=True,
    )