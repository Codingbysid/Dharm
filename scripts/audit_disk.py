"""Detailed Disk Inventory and Deduplication Audit for Dharm.

Audits:
- Project artifacts directories (artifacts-cpu-bs4, artifacts-v1.1-ablation-zeroshot, artifacts-v1.1, etc.)
- Hugging Face cache (~/.cache/huggingface/hub)
- Physical inodes, link counts (distinguishing hard links vs independent copies on macOS)
- SHA-256 hash verification of suspected duplicate weights
- Git tracking status via git ls-files
- Live app dependency verification
- Produces cleanup manifest with classification (KEEP, SAFE TO TRASH, REVIEW)
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

REPO_ROOT = Path(__file__).resolve().parent.parent
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

# Exact files loaded by the live app
LIVE_APP_FILES = {
    REPO_ROOT / "decision-corpus-ml" / "artifacts-v1.1" / "abstention_clf.joblib",
    REPO_ROOT / "decision-corpus-ml" / "artifacts-v1.6" / "pair_scorer_config.json",
    REPO_ROOT / "decision-corpus-ml" / "artifacts-v1.6" / "router_config.json",
    REPO_ROOT / "decision-corpus-ml" / "artifacts-v1.6" / "workflow_router.joblib",
    REPO_ROOT / "decision-corpus-ml" / "artifacts-v1.6" / "pair_verifier_model.joblib",
    REPO_ROOT / "decision-corpus-ml" / "artifacts-v1.6" / "verifier_config.json",
    REPO_ROOT / "decision-corpus-ml" / "artifacts-v1.6" / "rc1_manifest.json",
    REPO_ROOT / "decision-corpus-ml" / "artifacts-v1.6" / "rc1_parity_report.json",
}

# Hugging Face models actively used by the live app
LIVE_HF_MODELS = {
    "models--BAAI--bge-small-en-v1.5",
    "models--BAAI--bge-reranker-base",
    "models--cross-encoder--nli-deberta-v3-small",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def is_git_tracked(path: Path) -> bool:
    rel_path = path.relative_to(REPO_ROOT)
    cmd = ["git", "ls-files", "--error-unmatch", str(rel_path)]
    res = subprocess.run(cmd, cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return res.returncode == 0


def format_bytes(b: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024.0:
            return f"{b:.2f} {unit}"
        b /= 1024.0
    return f"{b:.2f} PB"


def main():
    print("=" * 80)
    print("Dharm Disk Usage Audit & Deduplication Manifest (macOS)")
    print("=" * 80)

    # 1. Project Large Files Inventory (> 10MB)
    print("\n1. Inventory of Large Files in Project (> 10MB):")
    project_large_files = []
    for root, dirs, files in os.walk(REPO_ROOT / "decision-corpus-ml"):
        for f in files:
            fp = Path(root) / f
            try:
                st = fp.stat()
                if st.st_size > 10 * 1024 * 1024:
                    project_large_files.append((fp, st))
            except OSError:
                pass

    project_large_files.sort(key=lambda x: -x[1].st_size)

    # Compute hashes and group by hash
    hash_map = {}
    table_rows = []

    for fp, st in project_large_files:
        f_hash = sha256_file(fp)
        tracked = is_git_tracked(fp)
        is_live = fp in LIVE_APP_FILES

        # Classification
        if is_live:
            classification = "KEEP"
            reason = "Loaded by live DharmRC1Engine"
        elif "artifacts-cpu-bs4" in str(fp):
            classification = "SAFE TO TRASH"
            reason = "Retired v1.0 fine-tuned weights (collapsed pretrained relevance head)"
        elif "artifacts-v1.1-ablation-zeroshot" in str(fp):
            classification = "SAFE TO TRASH"
            reason = "Retired v1.1 ablation weights (superseded by frozen v1.3/RC1)"
        elif "artifacts-v1.1" in str(fp):
            classification = "SAFE TO TRASH"
            reason = "Retired v1.1 fine-tuned weights (bi-encoder/cross-encoder retired; classifier kept)"
        else:
            classification = "REVIEW"
            reason = "Large binary file"

        if f_hash in hash_map:
            duplicate_of = hash_map[f_hash]
        else:
            hash_map[f_hash] = fp
            duplicate_of = None

        table_rows.append({
            "path": fp,
            "rel_path": str(fp.relative_to(REPO_ROOT)),
            "size_bytes": st.st_size,
            "size_str": format_bytes(st.st_size),
            "inode": st.st_ino,
            "nlink": st.st_nlink,
            "tracked": tracked,
            "is_live": is_live,
            "hash": f_hash,
            "duplicate_of": duplicate_of,
            "classification": classification,
            "reason": reason,
        })

    for r in table_rows:
        dup_str = f" [EXACT DUPLICATE of {r['duplicate_of'].name}]" if r["duplicate_of"] else ""
        print(f"  • {r['rel_path']}")
        print(f"    Size: {r['size_str']} | Inode: {r['inode']} | Links: {r['nlink']} | Git Tracked: {r['tracked']} | Live: {r['is_live']}")
        print(f"    SHA256: {r['hash'][:16]}...{dup_str}")
        print(f"    Classification: [{r['classification']}] - {r['reason']}")

    # 2. Hugging Face Cache Audit
    print("\n2. Hugging Face Hub Cache Inventory (~/.cache/huggingface/hub):")
    hf_rows = []
    if HF_CACHE.exists():
        for item in sorted(HF_CACHE.iterdir()):
            if item.is_dir() and item.name.startswith("models--"):
                # Calculate size
                size = sum(f.stat().st_size for f in item.glob("**/*") if f.is_file())
                model_name = item.name.replace("models--", "").replace("--", "/")
                is_active = item.name in LIVE_HF_MODELS
                classification = "KEEP" if is_active else "REVIEW"
                reason = "Actively used by live RC1 engine" if is_active else "Retired experiment model"
                hf_rows.append({
                    "name": model_name,
                    "folder": item.name,
                    "path": item,
                    "size_bytes": size,
                    "size_str": format_bytes(size),
                    "active": is_active,
                    "classification": classification,
                    "reason": reason,
                })

    hf_rows.sort(key=lambda x: -x["size_bytes"])
    for r in hf_rows:
        print(f"  • {r['name']:<40} | Size: {r['size_str']:<10} | Active: {str(r['active']):<5} | [{r['classification']}] - {r['reason']}")

    # 3. Cleanup Manifest Summary
    safe_to_trash = [r for r in table_rows if r["classification"] == "SAFE TO TRASH"]
    total_reclaimable_bytes = sum(r["size_bytes"] for r in safe_to_trash)

    print("\n" + "=" * 80)
    print("Approved Move-to-Trash Cleanup Manifest (Reversible):")
    print("=" * 80)
    for idx, r in enumerate(safe_to_trash, 1):
        print(f"{idx}. {r['rel_path']}")
        print(f"   Size: {r['size_str']} | SHA256: {r['hash'][:16]}...")
        print(f"   Reason: {r['reason']}")

    print("\n" + "-" * 80)
    print(f"TOTAL RECLAIMABLE SPACE: {format_bytes(total_reclaimable_bytes)} ({total_reclaimable_bytes:,} bytes)")
    print("-" * 80)

    # 4. Generate Safe Move-to-Trash Shell Script
    trash_script_path = REPO_ROOT / "scripts" / "safe_trash.sh"
    trash_lines = [
        "#!/usr/bin/env bash",
        "# Reversible Move-to-Trash script for approved Dharm candidate files",
        "# Moves files to ~/.Trash/dharm_cleanup_<timestamp>/",
        "set -euo pipefail",
        "",
        'TRASH_DIR="$HOME/.Trash/dharm_cleanup_$(date +%Y%m%d_%H%M%S)"',
        'mkdir -p "$TRASH_DIR"',
        'echo "Moving approved retired weights to $TRASH_DIR..."',
        "",
    ]
    for r in safe_to_trash:
        trash_lines.append(f'mkdir -p "$TRASH_DIR/{r["path"].parent.relative_to(REPO_ROOT)}"')
        trash_lines.append(f'mv "{r["path"]}" "$TRASH_DIR/{r["path"].relative_to(REPO_ROOT)}"')

    trash_lines.extend([
        "",
        'echo "Move completed successfully! Files can be restored from $TRASH_DIR if needed."',
        'echo "Trash was NOT emptied automatically."',
    ])

    trash_script_path.write_text("\n".join(trash_lines) + "\n", encoding="utf8")
    trash_script_path.chmod(0o755)
    print(f"\nGenerated reversible move-to-Trash script: {trash_script_path}")


if __name__ == "__main__":
    main()
