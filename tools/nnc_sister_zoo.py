#!/usr/bin/env python3
"""Inspect, inventory and generate compatibility reports for Samsung Sister/Rose .nnc models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nnc_inspect  # noqa: E402


def summarize_nnc(path: Path) -> dict:
    info = nnc_inspect.parse_nnc(path)
    meta = info.get("compiler_metadata", {})
    subgraphs = info.get("subgraphs", [])
    ncp_secs = info.get("ncp_sections", [])
    ncp = ncp_secs[0] if ncp_secs else {}
    return {
        "file": path.name,
        "path": str(path),
        "size_mb": round(info.get("file_size", 0) / (1024.0 * 1024.0), 2),
        "schema_ver": info.get("schema_version"),
        "soc_type": meta.get("SOC_TYPE", ""),
        "quant_dev": meta.get("QUANT_DEV", ""),
        "npuc_version": meta.get("NPUC_VERSION", ""),
        "ncp_version": meta.get("NCP_VERSION", ""),
        "framework": meta.get("FRAMEWORK", ""),
        "compiled_cmd": str(meta.get("COMPILED_COMMAND", "")).replace("\n", " "),
        "inputs": [
            {"name": t["name"], "shape": t["shape"], "type": t["type"]}
            for t in (subgraphs[0]["inputs"] if subgraphs else [])
        ],
        "outputs": [
            {"name": t["name"], "shape": t["shape"], "type": t["type"]}
            for t in (subgraphs[0]["outputs"] if subgraphs else [])
        ],
        "ncp_magic_ok": bool(ncp.get("magic2_valid", False)),
        "ncp_body_mb": round(ncp.get("body_size", 0) / (1024.0 * 1024.0), 2) if ncp else 0.0,
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python tools/nnc_sister_zoo.py <dir_or_file.nnc> [--json]")
        return 1
    as_json = "--json" in sys.argv
    targets: list[Path] = []
    for arg in sys.argv[1:]:
        if arg == "--json":
            continue
        p = Path(arg)
        if p.is_dir():
            targets.extend(sorted(p.glob("*.nnc")))
        elif p.is_file():
            targets.append(p)

    reports = [summarize_nnc(t) for t in targets]
    if as_json:
        print(json.dumps(reports, indent=2))
        return 0

    print("=== Samsung Exynos .nnc Sister & Rose Model Zoo ===")
    for r in reports:
        ins = ", ".join(f"{x['name']}{x['shape']}" for x in r["inputs"])
        outs = ", ".join(f"{x['name']}{x['shape']}" for x in r["outputs"])
        print(
            f"- {r['file']} ({r['size_mb']} MiB) | SOC={r['soc_type']} | "
            f"NPUC={r['npuc_version']} | NCP={r['ncp_version']} (magic={r['ncp_magic_ok']})"
        )
        print(f"    IN : {ins}")
        print(f"    OUT: {outs}")
        if r["compiled_cmd"]:
            print(f"    CMD: {r['compiled_cmd'][:140]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
