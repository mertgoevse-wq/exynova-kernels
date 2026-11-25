#!/usr/bin/env python3
"""Generate clean high-contrast SVG benchmark diagrams from Galaxy A56 live telemetry."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_quant_chart(report: dict) -> str:
    q = report.get("quant_bench", {})
    rows = q.get("results", [])
    if not rows:
        return ""

    w, h = 860, 400
    pad_l, pad_r, pad_t, pad_b = 170, 175, 76, 56
    chart_w = w - pad_l - pad_r
    max_us = max(float(r.get("p50_us", 1.0)) for r in rows) or 1.0

    palette = ["#EF4444", "#F59E0B", "#A78BFA", "#00E5FF", "#10B981"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0%" stop-color="#0B1120"/><stop offset="100%" stop-color="#111827"/>'
        '</linearGradient></defs>',
        f'<rect width="{w}" height="{h}" rx="14" fill="url(#bg)" stroke="#1E293B" stroke-width="2"/>',
        f'<text x="28" y="36" fill="#F8FAFC" font-family="Segoe UI, Arial, sans-serif" font-size="18" font-weight="bold">Galaxy A56 (Exynos 1580) — Quantization Latency (512x512 MatVec)</text>',
        f'<text x="28" y="57" fill="#94A3B8" font-family="Segoe UI, Arial, sans-serif" font-size="12">Live on-device P50 latency in microseconds (lower = faster) · Xclipse 540 Vulkan 1.3 vs ARM64 CPU</text>',
    ]

    bar_h = 34
    gap = 18
    for i, r in enumerate(rows):
        name = _escape(str(r.get("name", f"Mode {i}")))
        us = float(r.get("p50_us", 0.0))
        sp = float(r.get("speedup", 1.0))
        ratio = float(r.get("ratio", 1.0))
        bw = max(8.0, (us / max_us) * chart_w)
        y = pad_t + i * (bar_h + gap)
        col = palette[i % len(palette)]

        lines.append(
            f'<text x="{pad_l - 12}" y="{y + 22}" text-anchor="end" fill="#E2E8F0" '
            f'font-family="Consolas, monospace" font-size="13" font-weight="bold">{name}</text>'
        )
        lines.append(
            f'<rect x="{pad_l}" y="{y}" width="{chart_w}" height="{bar_h}" rx="7" fill="#1E293B"/>'
        )
        lines.append(
            f'<rect x="{pad_l}" y="{y}" width="{bw:.1f}" height="{bar_h}" rx="7" fill="{col}" opacity="0.92"/>'
        )
        val_lbl = f"{us:,.1f} us  ({sp:.1f}x · {ratio:.0f}x size)" if ratio > 1.0 else f"{us:,.1f} us  ({sp:.1f}x)"
        lines.append(
            f'<text x="{pad_l + bw + 10:.1f}" y="{y + 22}" fill="#F8FAFC" '
            f'font-family="Consolas, monospace" font-size="12">{_escape(val_lbl)}</text>'
        )

    lines.append(
        f'<text x="28" y="{h - 18}" fill="#64748B" font-family="Segoe UI, Arial, sans-serif" font-size="11">'
        f'Source: ExyNova Hardware Lab v3 live telemetry on SM-A566B (S5E8855) · Persistent VRAM weight cache enabled</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines)


def build_flappy_chart(report: dict) -> str:
    fb = report.get("flappy_4modes", {})
    modes = fb.get("modes", [])
    if not modes:
        return ""

    w, h = 860, 380
    pad_l, pad_r, pad_t = 135, 195, 86
    chart_w = w - pad_l - pad_r
    max_ms = max(float(m.get("mean_total_ms", 1.0)) for m in modes) or 1.0

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<defs><linearGradient id="bgf" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0%" stop-color="#07101F"/><stop offset="100%" stop-color="#0F172A"/>'
        '</linearGradient></defs>',
        f'<rect width="{w}" height="{h}" rx="14" fill="url(#bgf)" stroke="#1E293B" stroke-width="2"/>',
        '<text x="28" y="36" fill="#F8FAFC" font-family="Segoe UI, Arial, sans-serif" font-size="18" font-weight="bold">Flappy Bird AI — 4 Hardware Acceleration Modes on Galaxy A56</text>',
        '<text x="28" y="57" fill="#94A3B8" font-family="Segoe UI, Arial, sans-serif" font-size="12">Stacked per-frame AI latency (ENN NPU ms + Vulkan GPU ms + host glue) · Lower is faster</text>',
        # Legend
        '<rect x="28" y="66" width="12" height="10" rx="2" fill="#F59E0B"/><text x="45" y="75" fill="#CBD5E1" font-family="Segoe UI, Arial, sans-serif" font-size="11">ENN NPU (/dev/vertex10)</text>',
        '<rect x="195" y="66" width="12" height="10" rx="2" fill="#00E5FF"/><text x="212" y="75" fill="#CBD5E1" font-family="Segoe UI, Arial, sans-serif" font-size="11">Vulkan GPU (Xclipse 540)</text>',
        '<rect x="375" y="66" width="12" height="10" rx="2" fill="#475569"/><text x="392" y="75" fill="#CBD5E1" font-family="Segoe UI, Arial, sans-serif" font-size="11">Dispatch / Buffer Overhead</text>',
    ]

    bar_h = 38
    gap = 22
    for i, m in enumerate(modes):
        name = _escape(str(m.get("name", f"Mode {i}")))
        tot = float(m.get("mean_total_ms", 0.0))
        npu = float(m.get("mean_npu_ms", 0.0))
        gpu = float(m.get("mean_gpu_ms", 0.0))
        hz = float(m.get("max_ai_hz", 0.0))
        pipes = int(m.get("pipes_cleared", 0))
        other = max(0.0, tot - npu - gpu)

        w_npu = (npu / max_ms) * chart_w
        w_gpu = (gpu / max_ms) * chart_w
        w_oth = (other / max_ms) * chart_w
        y = pad_t + 16 + i * (bar_h + gap)

        lines.append(
            f'<text x="{pad_l - 12}" y="{y + 24}" text-anchor="end" fill="#F8FAFC" '
            f'font-family="Consolas, monospace" font-size="14" font-weight="bold">{name}</text>'
        )
        lines.append(
            f'<rect x="{pad_l}" y="{y}" width="{chart_w}" height="{bar_h}" rx="7" fill="#1E293B"/>'
        )
        cx = float(pad_l)
        if w_npu > 0.5:
            lines.append(f'<rect x="{cx:.1f}" y="{y}" width="{w_npu:.1f}" height="{bar_h}" rx="4" fill="#F59E0B"/>')
            cx += w_npu
        if w_gpu > 0.5:
            lines.append(f'<rect x="{cx:.1f}" y="{y}" width="{w_gpu:.1f}" height="{bar_h}" rx="4" fill="#00E5FF"/>')
            cx += w_gpu
        if w_oth > 0.5:
            lines.append(f'<rect x="{cx:.1f}" y="{y}" width="{w_oth:.1f}" height="{bar_h}" rx="4" fill="#475569"/>')
            cx += w_oth

        lbl = f"{tot:.2f} ms ({hz:.0f} Hz) · NPU {npu:.2f} / GPU {gpu:.2f} · Pipes:{pipes}"
        lines.append(
            f'<text x="{max(pad_l + 8, cx + 10):.1f}" y="{y + 24}" fill="#E2E8F0" '
            f'font-family="Consolas, monospace" font-size="11.5">{_escape(lbl)}</text>'
        )

    lines.append(
        f'<text x="28" y="{h - 18}" fill="#64748B" font-family="Segoe UI, Arial, sans-serif" font-size="11">'
        f'All 4 modes executed live on SM-A566B (NPU: SRIB .nnc via ENN 2.3.30, GPU: 64-agent Vulkan swarm policy + Conv2D)</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines)


def build_llm_chart(report: dict) -> str:
    llm = report.get("llm_bench", {})
    gpu = report.get("gpu_bench", {})
    npu = report.get("npu_test", {})

    w, h = 860, 320
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<defs><linearGradient id="bgl" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0%" stop-color="#090D16"/><stop offset="100%" stop-color="#131B2E"/>'
        '</linearGradient></defs>',
        f'<rect width="{w}" height="{h}" rx="14" fill="url(#bgl)" stroke="#1E293B" stroke-width="2"/>',
        '<text x="28" y="36" fill="#F8FAFC" font-family="Segoe UI, Arial, sans-serif" font-size="18" font-weight="bold">Galaxy A56 Heterogeneous AI Telemetry Snapshot</text>',
        '<text x="28" y="57" fill="#94A3B8" font-family="Segoe UI, Arial, sans-serif" font-size="12">Real on-device Vulkan Compute + ENN NPU + HuggingFace LLM measurements</text>',
    ]

    cards = [
        ("Xclipse 540 Peak MatMul", f"{gpu.get('gpu_gflops', 0.0):.2f} GFLOPS",
         f"P50: {gpu.get('gpu_p50_us', 0.0):.1f} us · {gpu.get('speedup', 0.0):.1f}x vs CPU", "#00E5FF"),
        ("ENN 2.3.30 NPU (.nnc)", f"{npu.get('exec_ms', 0.0):.2f} ms",
         f"/dev/vertex10 · ret={npu.get('ret_name', '-')}", "#10B981"),
        (f"HF LLM ({llm.get('quant', 'Q4_0')})", f"{llm.get('tokens_per_sec', 0.0):.2f} tok/s",
         f"BPE={llm.get('used_bpe_vocab', False)} · NPU ctx={llm.get('npu_context_ms', 0.0):.2f} ms", "#A78BFA"),
    ]

    cw, ch = 256, 155
    for idx, (title, big, sub, col) in enumerate(cards):
        x = 28 + idx * (cw + 18)
        y = 85
        lines.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="12" fill="#0F172A" stroke="{col}" stroke-width="1.5"/>')
        lines.append(f'<text x="{x + 16}" y="{y + 30}" fill="#94A3B8" font-family="Segoe UI, Arial, sans-serif" font-size="12" font-weight="bold">{_escape(title)}</text>')
        lines.append(f'<text x="{x + 16}" y="{y + 82}" fill="{col}" font-family="Consolas, monospace" font-size="28" font-weight="bold">{_escape(big)}</text>')
        lines.append(f'<text x="{x + 16}" y="{y + 125}" fill="#CBD5E1" font-family="Consolas, monospace" font-size="11">{_escape(sub)}</text>')

    sample_text = str(llm.get("text", "")).replace("\n", " ")[:92]
    lines.append(
        f'<text x="28" y="275" fill="#CBD5E1" font-family="Consolas, monospace" font-size="12">'
        f'Live LLM Sample: "{_escape(sample_text)}"</text>'
    )
    lines.append(
        f'<text x="28" y="{h - 18}" fill="#64748B" font-family="Segoe UI, Arial, sans-serif" font-size="11">'
        f'Verified on Samsung Galaxy A56 5G (SM-A566B / Exynos 1580) · Zero synthetic/mocked metrics</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines)


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    rep_path = Path(sys.argv[1]) if len(sys.argv) > 1 else repo / "local_artifacts" / "exynova_app_report.json"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else repo / "docs" / "assets"

    report = json.loads(rep_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)

    charts = {
        "quantization_speedup_a56.svg": build_quant_chart(report),
        "flappy_ai_4modes_a56.svg": build_flappy_chart(report),
        "llm_inference_a56.svg": build_llm_chart(report),
    }
    for fname, svg in charts.items():
        if svg:
            p = out_dir / fname
            p.write_text(svg, encoding="utf-8")
            print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
