#!/usr/bin/env python3
"""Download real open LLMs from HuggingFace and convert to ExyNova .exyllm v2 + .tok.

Supports:
  - tinystories-15m : karpathy/tinyllamas (stories15M.bin + tokenizer.bin)
  - tinystories-42m : karpathy/tinyllamas (stories42M.bin + tokenizer.bin)
  - smollm2-135m    : HuggingFaceTB/SmolLM2-135M-Instruct (safetensors + tokenizer.json)

Quantizations supported:
  - q4_0 (quant_type=2): 4-bit block-32 symmetric
  - q8_0 (quant_type=1): 8-bit block-32 symmetric
  - f16  (quant_type=3): IEEE-754 float16
  - f32  (quant_type=0): raw float32
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

QUANT_IDS = {
    "f32": 0,
    "q8_0": 1,
    "q4_0": 2,
    "f16": 3,
}


def quantize_q8_0_np(mat: np.ndarray) -> tuple[bytes, bytes]:
    rows, cols = mat.shape
    pad_cols = ((cols + 31) // 32) * 32
    if pad_cols != cols:
        padded = np.zeros((rows, pad_cols), dtype=np.float32)
        padded[:, :cols] = mat
    else:
        padded = np.ascontiguousarray(mat, dtype=np.float32)
    blocks = padded.reshape(rows, pad_cols // 32, 32)
    amax = np.max(np.abs(blocks), axis=2)
    scales = np.where(amax > 0.0, amax / 127.0, 0.0).astype(np.float32)
    inv_scales = np.where(scales > 0.0, 1.0 / scales, 0.0)[:, :, None]
    q = np.clip(np.rint(blocks * inv_scales), -127, 127).astype(np.int8)
    # 4 int8 per uint32/int32 in little-endian matches direct int8 byte order
    packed_bytes = q.reshape(-1).tobytes()
    return packed_bytes, scales.reshape(-1).tobytes()


def quantize_q4_0_np(mat: np.ndarray) -> tuple[bytes, bytes]:
    rows, cols = mat.shape
    pad_cols = ((cols + 31) // 32) * 32
    if pad_cols != cols:
        padded = np.zeros((rows, pad_cols), dtype=np.float32)
        padded[:, :cols] = mat
    else:
        padded = np.ascontiguousarray(mat, dtype=np.float32)
    blocks = padded.reshape(rows, pad_cols // 32, 32)
    amax = np.max(np.abs(blocks), axis=2)
    scales = np.where(amax > 0.0, amax / 7.0, 0.0).astype(np.float32)
    inv_scales = np.where(scales > 0.0, 1.0 / scales, 0.0)[:, :, None]
    q = np.clip(np.rint(blocks * inv_scales), -8, 7).astype(np.int32) + 8
    u4 = (q & 0xF).astype(np.uint32).reshape(-1, 8)
    words = np.zeros(u4.shape[0], dtype=np.uint32)
    for nib in range(8):
        words |= (u4[:, nib] << (nib * 4))
    return words.tobytes(), scales.reshape(-1).tobytes()


def quantize_f16_np(mat: np.ndarray) -> bytes:
    rows, cols = mat.shape
    pad_cols = ((cols + 1) // 2) * 2
    if pad_cols != cols:
        padded = np.zeros((rows, pad_cols), dtype=np.float32)
        padded[:, :cols] = mat
    else:
        padded = np.ascontiguousarray(mat, dtype=np.float32)
    return padded.astype(np.float16).tobytes()


def write_matrix_exyllm(fout, mat: np.ndarray, quant_name: str) -> None:
    qt = QUANT_IDS[quant_name]
    mat = np.ascontiguousarray(mat, dtype=np.float32)
    if qt == 0:
        fout.write(mat.tobytes())
    elif qt == 1:
        packed, scales = quantize_q8_0_np(mat)
        fout.write(packed)
        fout.write(scales)
    elif qt == 2:
        packed, scales = quantize_q4_0_np(mat)
        fout.write(packed)
        fout.write(scales)
    elif qt == 3:
        fout.write(quantize_f16_np(mat))
    else:
        raise ValueError(f"Unsupported quant: {quant_name}")


def write_exyllm_v2_header(
    fout,
    vocab_size: int,
    dim: int,
    hidden_dim: int,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    max_seq_len: int,
    rms_eps: float,
    rope_theta: float,
    quant_id: int,
    rope_mode: int,
    bos_id: int,
    eos_id: int,
) -> None:
    fout.write(b"ELLM")
    fout.write(
        struct.pack(
            "<8I2f4I",
            2,  # version = 2
            vocab_size,
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            max_seq_len,
            rms_eps,
            rope_theta,
            quant_id,
            rope_mode,
            bos_id,
            eos_id,
        )
    )


def convert_llama2c_tok_to_etok(tok_bin_path: Path, out_tok_path: Path, vocab_size: int) -> None:
    data = tok_bin_path.read_bytes()
    off = 4  # skip int32 max_token_length
    entries: list[tuple[float, bytes]] = []
    for _ in range(vocab_size):
        score, slen = struct.unpack_from("<fi", data, off)
        off += 8
        piece = data[off : off + slen]
        off += slen
        entries.append((score, piece))

    with out_tok_path.open("wb") as f:
        f.write(b"ETOK")
        f.write(struct.pack("<5I", 1, vocab_size, 1, 2, 2))  # style=2 raw UTF-8 llama2.c
        for score, piece in entries:
            f.write(struct.pack("<fI", float(score), len(piece)))
            f.write(piece)


def convert_hf_tokenizer_json_to_etok(
    tok_json_path: Path,
    out_tok_path: Path,
    vocab_size: int,
    bos_id: int,
    eos_id: int,
) -> None:
    obj = json.loads(tok_json_path.read_text(encoding="utf-8"))
    model = obj.get("model", {})
    vocab_map: dict[str, int] = dict(model.get("vocab", {}))
    for added in obj.get("added_tokens", []):
        vocab_map[added["content"]] = int(added["id"])

    merges = model.get("merges", [])
    merge_rank: dict[str, int] = {}
    for rank, m in enumerate(merges):
        if isinstance(m, list):
            key = "".join(m)
        else:
            key = m.replace(" ", "")
        if key not in merge_rank:
            merge_rank[key] = rank

    pieces: list[tuple[float, bytes]] = [(0.0, b"") for _ in range(vocab_size)]
    for piece_str, tid in vocab_map.items():
        if 0 <= tid < vocab_size:
            rank = merge_rank.get(piece_str)
            score = -float(rank) if rank is not None else -1000000.0
            pieces[tid] = (score, piece_str.encode("utf-8"))

    with out_tok_path.open("wb") as f:
        f.write(b"ETOK")
        f.write(struct.pack("<5I", 1, vocab_size, bos_id, eos_id, 1))  # style=1 byte-level BPE 'Ġ'
        for score, pbytes in pieces:
            f.write(struct.pack("<fI", float(score), len(pbytes)))
            f.write(pbytes)


def export_tinyllamas(bin_name: str, out_stem: str, quants: list[str], out_dir: Path) -> None:
    print(f"Downloading karpathy/tinyllamas/{bin_name} & llama2.c tokenizer.bin...")
    model_path = Path(hf_hub_download("karpathy/tinyllamas", bin_name))
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_path = out_dir / "_cache_llama2_tokenizer.bin"
    if not tok_path.is_file():
        urllib.request.urlretrieve(
            "https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin",
            str(tok_path),
        )

    raw = model_path.read_bytes()
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_raw, seq_len = struct.unpack_from("<7i", raw, 0)
    shared = vocab_raw > 0
    vocab_size = abs(vocab_raw)
    head_dim = dim // n_heads
    kv_dim = n_kv_heads * head_dim

    print(
        f"Loaded {bin_name}: dim={dim}, hidden={hidden_dim}, layers={n_layers}, "
        f"heads={n_heads}/{n_kv_heads}, vocab={vocab_size}, seq={seq_len}, shared={shared}"
    )

    weights = np.frombuffer(raw, dtype=np.float32, offset=28)
    cursor = 0

    def take(shape: tuple[int, ...]) -> np.ndarray:
        nonlocal cursor
        cnt = int(np.prod(shape))
        arr = weights[cursor : cursor + cnt].reshape(shape)
        cursor += cnt
        return arr

    emb = take((vocab_size, dim))
    rms_att = take((n_layers, dim))
    wq = take((n_layers, dim, dim))
    wk = take((n_layers, kv_dim, dim))
    wv = take((n_layers, kv_dim, dim))
    wo = take((n_layers, dim, dim))
    rms_ffn = take((n_layers, dim))
    w1 = take((n_layers, hidden_dim, dim))
    w2 = take((n_layers, dim, hidden_dim))
    w3 = take((n_layers, hidden_dim, dim))
    rms_final = take((dim,))
    _ = take((seq_len, head_dim // 2))  # freq_cis_real
    _ = take((seq_len, head_dim // 2))  # freq_cis_imag
    wcls = emb if shared else take((vocab_size, dim))

    out_dir.mkdir(parents=True, exist_ok=True)
    for q in quants:
        exy_path = out_dir / f"{out_stem}_{q}.exyllm"
        tok_out = Path(str(exy_path) + ".tok")
        with exy_path.open("wb") as f:
            write_exyllm_v2_header(
                f,
                vocab_size=vocab_size,
                dim=dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                max_seq_len=min(seq_len, 256),
                rms_eps=1e-5,
                rope_theta=10000.0,
                quant_id=QUANT_IDS[q],
                rope_mode=0,
                bos_id=1,
                eos_id=2,
            )
            f.write(emb.astype(np.float32).tobytes())
            for l in range(n_layers):
                f.write(rms_att[l].astype(np.float32).tobytes())
                write_matrix_exyllm(f, wq[l], q)
                write_matrix_exyllm(f, wk[l], q)
                write_matrix_exyllm(f, wv[l], q)
                write_matrix_exyllm(f, wo[l], q)
                f.write(rms_ffn[l].astype(np.float32).tobytes())
                write_matrix_exyllm(f, w1[l], q)
                write_matrix_exyllm(f, w2[l], q)
                write_matrix_exyllm(f, w3[l], q)
            f.write(rms_final.astype(np.float32).tobytes())
            write_matrix_exyllm(f, wcls, q)

        convert_llama2c_tok_to_etok(tok_path, tok_out, vocab_size)
        print(f"Wrote {exy_path} ({exy_path.stat().st_size / (1024*1024):.2f} MiB) + {tok_out.name}")


def export_smollm2_135m(quants: list[str], out_dir: Path) -> None:
    from safetensors import safe_open

    repo = "HuggingFaceTB/SmolLM2-135M-Instruct"
    print(f"Downloading {repo}...")
    cfg_path = Path(hf_hub_download(repo, "config.json"))
    tok_json_path = Path(hf_hub_download(repo, "tokenizer.json"))
    st_path = Path(hf_hub_download(repo, "model.safetensors"))

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    vocab_size = int(cfg["vocab_size"])
    dim = int(cfg["hidden_size"])
    hidden_dim = int(cfg["intermediate_size"])
    n_layers = int(cfg["num_hidden_layers"])
    n_heads = int(cfg["num_attention_heads"])
    n_kv_heads = int(cfg.get("num_key_value_heads", n_heads))
    rms_eps = float(cfg.get("rms_norm_eps", 1e-5))
    rope_theta = float(cfg.get("rope_theta", 100000.0))
    bos_id = int(cfg.get("bos_token_id", 1))
    eos_id = int(cfg.get("eos_token_id", 2))

    print(
        f"Loaded SmolLM2-135M: dim={dim}, hidden={hidden_dim}, layers={n_layers}, "
        f"heads={n_heads}/{n_kv_heads}, vocab={vocab_size}, rope_theta={rope_theta}"
    )

    def to_f32(t) -> np.ndarray:
        if t.dtype != np.float32:
            # Handle torch/bfloat16/float16 returned via numpy or bytes
            return np.asarray(t, dtype=np.float32)
        return t

    # safetensors numpy framework converts bfloat16 if torch isn't installed? Let's load via raw bf16 if needed
    def get_tensor_f32(sf, name: str) -> np.ndarray:
        sl = sf.get_slice(name)
        dt = sl.get_dtype()
        if dt == "BF16":
            raw_u16 = sf.get_tensor(name).view(np.uint16)
            u32 = raw_u16.astype(np.uint32) << 16
            return u32.view(np.float32)
        return sf.get_tensor(name).astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    with safe_open(str(st_path), framework="np") as sf:
        emb = get_tensor_f32(sf, "model.embed_tokens.weight")
        rms_final = get_tensor_f32(sf, "model.norm.weight")
        wcls = (
            get_tensor_f32(sf, "lm_head.weight")
            if "lm_head.weight" in sf.keys()
            else emb
        )

        for q in quants:
            exy_path = out_dir / f"smollm2_135m_{q}.exyllm"
            tok_out = Path(str(exy_path) + ".tok")
            with exy_path.open("wb") as f:
                write_exyllm_v2_header(
                    f,
                    vocab_size=vocab_size,
                    dim=dim,
                    hidden_dim=hidden_dim,
                    n_layers=n_layers,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    max_seq_len=256,
                    rms_eps=rms_eps,
                    rope_theta=rope_theta,
                    quant_id=QUANT_IDS[q],
                    rope_mode=1,  # NeoX half-split RoPE
                    bos_id=bos_id,
                    eos_id=eos_id,
                )
                f.write(emb.astype(np.float32).tobytes())
                for l in range(n_layers):
                    pfx = f"model.layers.{l}"
                    f.write(get_tensor_f32(sf, f"{pfx}.input_layernorm.weight").tobytes())
                    write_matrix_exyllm(f, get_tensor_f32(sf, f"{pfx}.self_attn.q_proj.weight"), q)
                    write_matrix_exyllm(f, get_tensor_f32(sf, f"{pfx}.self_attn.k_proj.weight"), q)
                    write_matrix_exyllm(f, get_tensor_f32(sf, f"{pfx}.self_attn.v_proj.weight"), q)
                    write_matrix_exyllm(f, get_tensor_f32(sf, f"{pfx}.self_attn.o_proj.weight"), q)
                    f.write(get_tensor_f32(sf, f"{pfx}.post_attention_layernorm.weight").tobytes())
                    write_matrix_exyllm(f, get_tensor_f32(sf, f"{pfx}.mlp.gate_proj.weight"), q)
                    write_matrix_exyllm(f, get_tensor_f32(sf, f"{pfx}.mlp.down_proj.weight"), q)
                    write_matrix_exyllm(f, get_tensor_f32(sf, f"{pfx}.mlp.up_proj.weight"), q)
                f.write(rms_final.astype(np.float32).tobytes())
                write_matrix_exyllm(f, wcls, q)

            convert_hf_tokenizer_json_to_etok(tok_json_path, tok_out, vocab_size, bos_id, eos_id)
            print(f"Wrote {exy_path} ({exy_path.stat().st_size / (1024*1024):.2f} MiB) + {tok_out.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Download open HuggingFace LLMs and convert to .exyllm + .tok")
    ap.add_argument(
        "--preset",
        choices=["tinystories-15m", "tinystories-42m", "smollm2-135m"],
        default="tinystories-15m",
    )
    ap.add_argument(
        "--quant",
        nargs="+",
        choices=["q4_0", "q8_0", "f16", "f32"],
        default=["q4_0", "q8_0"],
    )
    ap.add_argument("--out-dir", type=Path, default=Path("local_artifacts/models"))
    args = ap.parse_args()

    if args.preset == "tinystories-15m":
        export_tinyllamas("stories15M.bin", "stories15m", args.quant, args.out_dir)
    elif args.preset == "tinystories-42m":
        export_tinyllamas("stories42M.bin", "stories42m", args.quant, args.out_dir)
    elif args.preset == "smollm2-135m":
        export_smollm2_135m(args.quant, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
