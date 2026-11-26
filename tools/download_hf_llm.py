#!/usr/bin/env python3
"""Download real open LLMs from HuggingFace and convert to ExyNova .exyllm (v2/v3) + .tok.

Presets:
  - tinystories-15m : karpathy/tinyllamas (stories15M.bin + llama2.c tokenizer.bin)
  - tinystories-42m : karpathy/tinyllamas (stories42M.bin + llama2.c tokenizer.bin)
  - smollm2-135m    : HuggingFaceTB/SmolLM2-135M-Instruct (ChatML BPE, NeoX RoPE)
  - qwen2.5-0.5b    : Qwen/Qwen2.5-0.5B-Instruct (ChatML BPE, NeoX RoPE, QKV bias, v3)
"""

from __future__ import annotations

import argparse
import json
import struct
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
    return q.reshape(-1).tobytes(), scales.reshape(-1).tobytes()


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


def write_exyllm_v3_header(
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
    has_qkv_bias: int = 0,
    alt_eos_id: int = 0,
) -> None:
    fout.write(b"ELLM")
    fout.write(
        struct.pack(
            "<8I2f6I",
            3,  # version = 3
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
            has_qkv_bias,  # bitmask: 1=qkv_bias, 2=fp16_emb, 4=q8_lm_head
            alt_eos_id,
        )
    )


def convert_llama2c_tok_to_etok(tok_bin_path: Path, out_tok_path: Path, vocab_size: int) -> None:
    data = tok_bin_path.read_bytes()
    off = 4
    entries: list[tuple[float, bytes]] = []
    for _ in range(vocab_size):
        score, slen = struct.unpack_from("<fi", data, off)
        off += 8
        piece = data[off : off + slen]
        off += slen
        entries.append((score, piece))

    with out_tok_path.open("wb") as f:
        f.write(b"ETOK")
        f.write(struct.pack("<5I", 1, vocab_size, 1, 2, 2))
        for score, piece in entries:
            f.write(struct.pack("<fI", float(score), len(piece)))
            f.write(piece)


def _gpt2_unicode_to_bytes() -> dict[str, bytes]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): bytes([b]) for b, c in zip(bs, cs)}


def _decode_gpt2_piece(s: str, u2b: dict[str, bytes], is_added: bool) -> bytes:
    if is_added:
        return s.encode("utf-8")
    out = bytearray()
    for ch in s:
        b = u2b.get(ch)
        if b is not None:
            out.extend(b)
        else:
            out.extend(ch.encode("utf-8"))
    return bytes(out)


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
    added_set: set[str] = set()
    for added in obj.get("added_tokens", []):
        content = added["content"]
        vocab_map[content] = int(added["id"])
        added_set.add(content)

    u2b = _gpt2_unicode_to_bytes()
    merges = model.get("merges", [])
    merge_rank: dict[bytes, int] = {}
    for rank, m in enumerate(merges):
        m_str = "".join(m) if isinstance(m, list) else m.replace(" ", "")
        m_bytes = _decode_gpt2_piece(m_str, u2b, False)
        if m_bytes not in merge_rank:
            merge_rank[m_bytes] = rank

    pieces: list[tuple[float, bytes]] = [(0.0, b"") for _ in range(vocab_size)]
    for piece_str, tid in vocab_map.items():
        if 0 <= tid < vocab_size:
            pbytes = _decode_gpt2_piece(piece_str, u2b, piece_str in added_set)
            rank = merge_rank.get(pbytes)
            score = -float(rank) if rank is not None else -1000000.0
            pieces[tid] = (score, pbytes)

    with out_tok_path.open("wb") as f:
        f.write(b"ETOK")
        # tok_style=2 (raw decoded UTF-8 byte-level BPE)
        f.write(struct.pack("<5I", 1, vocab_size, bos_id, eos_id, 2))
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
    _ = take((seq_len, head_dim // 2))
    _ = take((seq_len, head_dim // 2))
    wcls = emb if shared else take((vocab_size, dim))

    for q in quants:
        exy_path = out_dir / f"{out_stem}_{q}.exyllm"
        tok_out = Path(str(exy_path) + ".tok")
        with exy_path.open("wb") as f:
            write_exyllm_v3_header(
                f, vocab_size, dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                min(seq_len, 256), 1e-5, 10000.0, QUANT_IDS[q], 0, 1, 2, 0, 0,
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


def export_hf_causal_lm(
    repo: str,
    out_stem: str,
    quants: list[str],
    out_dir: Path,
    default_bos: int,
    default_eos: int,
    alt_eos: int,
) -> None:
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
    rms_eps = float(cfg.get("rms_norm_eps", 1e-6))
    rope_theta = float(cfg.get("rope_theta", 100000.0))
    bos_id = int(cfg.get("bos_token_id") if cfg.get("bos_token_id") is not None else default_bos)
    eos_id = int(cfg.get("eos_token_id") if cfg.get("eos_token_id") is not None else default_eos)

    out_dir.mkdir(parents=True, exist_ok=True)
    with st_path.open("rb") as st_f:
        hdr_len = struct.unpack("<Q", st_f.read(8))[0]
        hdr = json.loads(st_f.read(hdr_len).decode("utf-8"))
        data_base = 8 + hdr_len
        keys = {k for k in hdr.keys() if k != "__metadata__"}

        def get_tensor_f32(name: str) -> np.ndarray:
            meta = hdr[name]
            dt = meta["dtype"]
            shape = tuple(meta["shape"])
            start, end = meta["data_offsets"]
            st_f.seek(data_base + start)
            raw = st_f.read(end - start)
            if dt == "BF16":
                u16 = np.frombuffer(raw, dtype="<u2").reshape(shape)
                return (u16.astype(np.uint32) << 16).view(np.float32)
            if dt == "F16":
                return np.frombuffer(raw, dtype="<f2").reshape(shape).astype(np.float32)
            if dt == "F32":
                return np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
            raise ValueError(f"Unsupported safetensors dtype {dt} for {name}")

        has_qkv_bias = 1 if "model.layers.0.self_attn.q_proj.bias" in keys else 0
        print(
            f"Loaded {repo}: dim={dim}, hidden={hidden_dim}, layers={n_layers}, "
            f"heads={n_heads}/{n_kv_heads}, vocab={vocab_size}, qkv_bias={bool(has_qkv_bias)}"
        )

        emb = get_tensor_f32("model.embed_tokens.weight")
        rms_final = get_tensor_f32("model.norm.weight")
        wcls = get_tensor_f32("lm_head.weight") if "lm_head.weight" in keys else emb

        for q in quants:
            exy_path = out_dir / f"{out_stem}_{q}.exyllm"
            tok_out = Path(str(exy_path) + ".tok")
            # Compact v3 bitmask: bit0=qkv_bias, bit1=fp16_emb (50% smaller RAM), bit2=q8_0 lm_head when q==q4_0
            flags = has_qkv_bias | 2 | (4 if q == "q4_0" else 0)
            head_q = "q8_0" if q == "q4_0" else q
            with exy_path.open("wb") as f:
                write_exyllm_v3_header(
                    f, vocab_size, dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                    512, rms_eps, rope_theta, QUANT_IDS[q], 1, bos_id, eos_id,
                    flags, alt_eos,
                )
                f.write(emb.astype(np.float16).tobytes())
                for l in range(n_layers):
                    pfx = f"model.layers.{l}"
                    f.write(get_tensor_f32(f"{pfx}.input_layernorm.weight").tobytes())
                    write_matrix_exyllm(f, get_tensor_f32(f"{pfx}.self_attn.q_proj.weight"), q)
                    write_matrix_exyllm(f, get_tensor_f32(f"{pfx}.self_attn.k_proj.weight"), q)
                    write_matrix_exyllm(f, get_tensor_f32(f"{pfx}.self_attn.v_proj.weight"), q)
                    if has_qkv_bias:
                        f.write(get_tensor_f32(f"{pfx}.self_attn.q_proj.bias").tobytes())
                        f.write(get_tensor_f32(f"{pfx}.self_attn.k_proj.bias").tobytes())
                        f.write(get_tensor_f32(f"{pfx}.self_attn.v_proj.bias").tobytes())
                    write_matrix_exyllm(f, get_tensor_f32(f"{pfx}.self_attn.o_proj.weight"), q)
                    f.write(get_tensor_f32(f"{pfx}.post_attention_layernorm.weight").tobytes())
                    write_matrix_exyllm(f, get_tensor_f32(f"{pfx}.mlp.gate_proj.weight"), q)
                    write_matrix_exyllm(f, get_tensor_f32(f"{pfx}.mlp.down_proj.weight"), q)
                    write_matrix_exyllm(f, get_tensor_f32(f"{pfx}.mlp.up_proj.weight"), q)
                f.write(rms_final.astype(np.float32).tobytes())
                write_matrix_exyllm(f, wcls, head_q)

            convert_hf_tokenizer_json_to_etok(tok_json_path, tok_out, vocab_size, bos_id, eos_id)
            print(f"Wrote {exy_path} ({exy_path.stat().st_size / (1024*1024):.2f} MiB) + {tok_out.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Download open HuggingFace LLMs and convert to .exyllm v3 + .tok")
    ap.add_argument(
        "--preset",
        choices=["tinystories-15m", "tinystories-42m", "smollm2-135m", "qwen2.5-0.5b", "qwen2.5-1.5b"],
        default="smollm2-135m",
    )
    ap.add_argument(
        "--quant",
        nargs="+",
        choices=["q4_0", "q8_0", "f16", "f32"],
        default=["q4_0"],
    )
    ap.add_argument("--out-dir", type=Path, default=Path("local_artifacts/models"))
    args = ap.parse_args()

    if args.preset == "tinystories-15m":
        export_tinyllamas("stories15M.bin", "stories15m", args.quant, args.out_dir)
    elif args.preset == "tinystories-42m":
        export_tinyllamas("stories42M.bin", "stories42m", args.quant, args.out_dir)
    elif args.preset == "smollm2-135m":
        export_hf_causal_lm("HuggingFaceTB/SmolLM2-135M-Instruct", "smollm2_135m",
                            args.quant, args.out_dir, 1, 2, 0)
    elif args.preset == "qwen2.5-0.5b":
        export_hf_causal_lm("Qwen/Qwen2.5-0.5B-Instruct", "qwen25_05b",
                            args.quant, args.out_dir, 151643, 151645, 151643)
    elif args.preset == "qwen2.5-1.5b":
        export_hf_causal_lm("Qwen/Qwen2.5-1.5B-Instruct", "qwen25_15b",
                            args.quant, args.out_dir, 151643, 151645, 151643)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
