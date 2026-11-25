#!/usr/bin/env python3
"""Inspect Samsung ENN (.nnc / ENNC FlatBuffer) model containers.

Parses the TFLite-v3-derived ENNC FlatBuffer container, embedded FlexBuffer
compiler metadata (ENN_NPUOptions), tensor descriptors, operator codes, and the
embedded Samsung NCP binary header without external dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any

# Standard TFLite TensorType enum (subset + common quantized types)
TENSOR_TYPES: dict[int, str] = {
    0: "FLOAT32",
    1: "FLOAT16",
    2: "INT32",
    3: "UINT8",
    4: "INT64",
    5: "STRING",
    6: "BOOL",
    7: "INT16",
    8: "COMPLEX64",
    9: "INT8",
    10: "FLOAT64",
    11: "COMPLEX128",
    12: "UINT64",
    13: "RESOURCE",
    14: "VARIANT",
    15: "UINT32",
    16: "UINT16",
    17: "INT4",
    18: "BFLOAT16",
}

# TFLite BuiltinOperator codes observed in ENNC + common TFLite ops
BUILTIN_OPERATORS: dict[int, str] = {
    6: "DEQUANTIZE",
    25: "RESHAPE",
    53: "TRANSPOSE",
    89: "SQUEEZE",
    114: "QUANTIZE",
    1012: "ENN_PREPROC_OR_CUSTOM_1012",
    1013: "ENN_NPU",
}

BUILTIN_OPTIONS_TYPES: dict[int, str] = {
    0: "NONE",
    37: "TransposeOptions",
    38: "DequantizeOptions",
    89: "QuantizeOptions",
    126: "ENN_Options_126",
    127: "ENN_NPUOptions",
}

NCP_MAGIC1 = 0x0C0FFEE0
NCP_MAGIC2 = 0xC0DEC0DE


class Ansi:
    """Minimal ANSI color helper."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def wrap(self, code: str, text: str) -> str:
        if not self.enabled:
            return text
        return f"\033[{code}m{text}\033[0m"

    def bold(self, text: str) -> str:
        return self.wrap("1", text)

    def cyan(self, text: str) -> str:
        return self.wrap("36", text)

    def green(self, text: str) -> str:
        return self.wrap("32", text)

    def yellow(self, text: str) -> str:
        return self.wrap("33", text)

    def magenta(self, text: str) -> str:
        return self.wrap("35", text)

    def dim(self, text: str) -> str:
        return self.wrap("2", text)


class FlatBufferReader:
    """Low-level zero-copy FlatBuffer reader."""

    def __init__(self, data: bytes) -> None:
        self.data = data

    def u8(self, off: int) -> int:
        return self.data[off]

    def i8(self, off: int) -> int:
        return struct.unpack_from("<b", self.data, off)[0]

    def u16(self, off: int) -> int:
        return struct.unpack_from("<H", self.data, off)[0]

    def u32(self, off: int) -> int:
        return struct.unpack_from("<I", self.data, off)[0]

    def i32(self, off: int) -> int:
        return struct.unpack_from("<i", self.data, off)[0]

    def u64(self, off: int) -> int:
        return struct.unpack_from("<Q", self.data, off)[0]

    def vtable(self, table_off: int) -> list[int]:
        vt_off = table_off - self.i32(table_off)
        vsize = self.u16(vt_off)
        count = max(0, (vsize - 4) // 2)
        return [self.u16(vt_off + 4 + 2 * i) for i in range(count)]

    def field_off(self, table_off: int, fields: list[int], idx: int) -> int | None:
        if idx >= len(fields) or fields[idx] == 0:
            return None
        return table_off + fields[idx]

    def deref(self, off: int) -> int:
        return off + self.u32(off)

    def read_str(self, field_pos: int | None) -> str | None:
        if field_pos is None:
            return None
        s_off = self.deref(field_pos)
        length = self.u32(s_off)
        return self.data[s_off + 4 : s_off + 4 + length].decode("utf-8", "replace")

    def read_i32_vec(self, field_pos: int | None) -> list[int]:
        if field_pos is None:
            return []
        v_off = self.deref(field_pos)
        count = self.u32(v_off)
        return [self.i32(v_off + 4 + 4 * i) for i in range(count)]

    def read_u8_vec(self, field_pos: int | None) -> tuple[int, int]:
        """Return (payload_offset, length) for a byte vector."""
        if field_pos is None:
            return (0, 0)
        v_off = self.deref(field_pos)
        length = self.u32(v_off)
        return (v_off + 4, length)


def parse_flexbuffer_map(buf: bytes) -> dict[str, Any]:
    """Parse a FlatBuffers FlexBuffer root Map into a Python dict."""
    if len(buf) < 3:
        return {}
    byte_width = buf[-1]
    packed_type = buf[-2]
    root_type = packed_type >> 2
    root_bit_width = packed_type & 3
    w = 1 << root_bit_width
    # FlexBuffer Type::FBT_MAP == 9
    if root_type != 9 or byte_width not in (1, 2, 4, 8):
        return {}
    root_off = len(buf) - 2 - byte_width
    if root_off < 0:
        return {}
    rel = int.from_bytes(buf[root_off : root_off + byte_width], "little")
    map_start = root_off - rel
    if map_start < 3 * w:
        return {}

    keys_rel = int.from_bytes(buf[map_start - 3 * w : map_start - 2 * w], "little")
    keys_vec = (map_start - 3 * w) - keys_rel
    keys_w = int.from_bytes(buf[map_start - 2 * w : map_start - w], "little")
    count = int.from_bytes(buf[map_start - w : map_start], "little")
    if keys_w not in (1, 2, 4, 8):
        return {}

    out: dict[str, Any] = {}
    for i in range(count):
        koff_pos = keys_vec + i * keys_w
        krel = int.from_bytes(buf[koff_pos : koff_pos + keys_w], "little")
        kpos = koff_pos - krel
        kend = buf.find(b"\x00", kpos)
        if kend == -1:
            continue
        key = buf[kpos:kend].decode("utf-8", "replace")

        vpos = map_start + i * w
        pt = buf[map_start + count * w + i]
        vt = pt >> 2
        vw = 1 << (pt & 3)
        if vt == 5:  # FBT_STRING
            vrel = int.from_bytes(buf[vpos : vpos + w], "little")
            spos = vpos - vrel
            slen = int.from_bytes(buf[spos - vw : spos], "little")
            val: Any = buf[spos : spos + slen].decode("utf-8", "replace")
        elif vt == 1:  # FBT_INT
            val = int.from_bytes(buf[vpos : vpos + w], "little", signed=True)
        elif vt == 2:  # FBT_UINT
            val = int.from_bytes(buf[vpos : vpos + w], "little", signed=False)
        elif vt == 3:  # FBT_FLOAT
            fmt = "<f" if w == 4 else "<d"
            val = struct.unpack_from(fmt, buf, vpos)[0] if w in (4, 8) else buf[vpos : vpos + w].hex()
        elif vt == 26:  # FBT_BOOL
            val = bool(int.from_bytes(buf[vpos : vpos + w], "little"))
        else:
            val = f"<flex_type={vt}:{buf[vpos : vpos + w].hex()}>"
        out[key] = val
    return out


def parse_ncp_header(data: bytes, offset: int, size: int) -> dict[str, Any] | None:
    """Parse Samsung NCP v26/v27 binary header at the given file offset."""
    if offset < 0 or offset + 0xC8 > len(data) or size < 0xC8:
        return None
    magic1 = struct.unpack_from("<I", data, offset)[0]
    if magic1 != NCP_MAGIC1:
        return None

    hdr_version, hdr_size, intrinsic_ver, abi_ver = struct.unpack_from("<IIII", data, offset + 4)
    model_name = data[offset + 20 : offset + 40].split(b"\x00", 1)[0].decode("ascii", "replace")
    (
        priority,
        flags,
        periodicity,
        workload,
        comp_workload,
        io_workload,
        addr_vec_off,
        addr_vec_cnt,
        mem_vec_off,
        mem_vec_cnt,
        pwrest_vec_off,
        pwrest_vec_cnt,
        intr_vec_off,
        intr_vec_cnt,
        grp_vec_off,
        grp_vec_cnt,
        thr_vec_off,
        thr_vec_cnt,
        body_ver,
        body_off,
        body_sz,
        io_vec_off,
        io_vec_cnt,
        rq_vec_off,
        rq_vec_sz,
        unique_id,
        llc_vec_off,
        llc_vec_cnt,
        fence_flag,
    ) = struct.unpack_from("<29I", data, offset + 40)
    magic2 = struct.unpack_from("<I", data, offset + 0xC4)[0]

    return {
        "file_offset": offset,
        "buffer_size": size,
        "magic1": f"0x{magic1:08x}",
        "magic2": f"0x{magic2:08x}",
        "magic2_valid": magic2 == NCP_MAGIC2,
        "hdr_version": hdr_version,
        "hdr_size": hdr_size,
        "intrinsic_version": f"0x{intrinsic_ver:08x}",
        "abi_version": f"0x{abi_ver:08x}",
        "model_name": model_name,
        "workload_cycles": workload,
        "computational_workload_kops": comp_workload,
        "io_workload_kb": io_workload,
        "address_vector_cnt": addr_vec_cnt,
        "memory_vector_cnt": mem_vec_cnt,
        "pwrest_vector_cnt": pwrest_vec_cnt,
        "interruption_vector_cnt": intr_vec_cnt,
        "group_vector_cnt": grp_vec_cnt,
        "thread_vector_cnt": thr_vec_cnt,
        "io_vector_cnt": io_vec_cnt,
        "llc_vector_cnt": llc_vec_cnt,
        "body_version": body_ver,
        "body_offset": body_off,
        "body_size": body_sz,
        "unique_id": unique_id,
    }


def parse_nnc(path: Path) -> dict[str, Any]:
    """Parse an ENNC FlatBuffer .nnc model container."""
    data = path.read_bytes()
    if len(data) < 16:
        raise ValueError(f"{path}: file too small ({len(data)} bytes)")
    magic = data[4:8]
    if magic != b"ENNC":
        raise ValueError(f"{path}: invalid magic {magic!r}, expected b'ENNC'")

    fb = FlatBufferReader(data)
    root = fb.u32(0)
    rfields = fb.vtable(root)

    version = fb.u32(fb.field_off(root, rfields, 0)) if fb.field_off(root, rfields, 0) else 0
    description = fb.read_str(fb.field_off(root, rfields, 3)) or ""

    # Parse buffers table (inline data vector OR external offset/size fields)
    buffers: list[dict[str, Any]] = []
    bvec_pos = fb.field_off(root, rfields, 4)
    if bvec_pos is not None:
        bvec = fb.deref(bvec_pos)
        nbufs = fb.u32(bvec)
        for i in range(nbufs):
            bt = fb.deref(bvec + 4 + 4 * i)
            bf = fb.vtable(bt)
            data_pos = fb.field_off(bt, bf, 0)
            if data_pos is not None:
                pay_off, pay_len = fb.read_u8_vec(data_pos)
            else:
                off_pos = fb.field_off(bt, bf, 1)
                sz_pos = fb.field_off(bt, bf, 2)
                pay_off = fb.u64(off_pos) if off_pos is not None else 0
                pay_len = fb.u64(sz_pos) if sz_pos is not None else 0
            buffers.append({"index": i, "offset": pay_off, "size": pay_len})

    # Parse operator_codes
    opcodes: list[dict[str, Any]] = []
    oc_pos = fb.field_off(root, rfields, 1)
    if oc_pos is not None:
        ocvec = fb.deref(oc_pos)
        noc = fb.u32(ocvec)
        for i in range(noc):
            ot = fb.deref(ocvec + 4 + 4 * i)
            of = fb.vtable(ot)
            dep_code = fb.i8(fb.field_off(ot, of, 0)) if fb.field_off(ot, of, 0) else 0
            custom = fb.read_str(fb.field_off(ot, of, 1))
            op_ver = fb.i32(fb.field_off(ot, of, 2)) if fb.field_off(ot, of, 2) else 1
            bcode = fb.i32(fb.field_off(ot, of, 3)) if fb.field_off(ot, of, 3) else dep_code
            name = custom if custom else BUILTIN_OPERATORS.get(bcode, f"OP_{bcode}")
            opcodes.append(
                {
                    "index": i,
                    "builtin_code": bcode,
                    "custom_code": custom,
                    "version": op_ver,
                    "name": name,
                }
            )

    # Parse subgraphs, tensors, and ENN operator metadata
    subgraphs: list[dict[str, Any]] = []
    compiler_metadata: dict[str, Any] = {}
    ncp_sections: list[dict[str, Any]] = []

    sg_pos = fb.field_off(root, rfields, 2)
    if sg_pos is not None:
        sgvec = fb.deref(sg_pos)
        nsg = fb.u32(sgvec)
        for si in range(nsg):
            sg = fb.deref(sgvec + 4 + 4 * si)
            sf = fb.vtable(sg)
            sg_name = fb.read_str(fb.field_off(sg, sf, 4)) or f"subgraph_{si}"
            in_indices = fb.read_i32_vec(fb.field_off(sg, sf, 1))
            out_indices = fb.read_i32_vec(fb.field_off(sg, sf, 2))

            tensors: list[dict[str, Any]] = []
            t_pos = fb.field_off(sg, sf, 0)
            if t_pos is not None:
                tvec = fb.deref(t_pos)
                nt = fb.u32(tvec)
                for ti in range(nt):
                    tt = fb.deref(tvec + 4 + 4 * ti)
                    tf = fb.vtable(tt)
                    shape = fb.read_i32_vec(fb.field_off(tt, tf, 0))
                    t_type_id = fb.u8(fb.field_off(tt, tf, 1)) if fb.field_off(tt, tf, 1) else 0
                    buf_idx = fb.u32(fb.field_off(tt, tf, 2)) if fb.field_off(tt, tf, 2) else 0
                    t_name = fb.read_str(fb.field_off(tt, tf, 3)) or f"tensor_{ti}"
                    buf_entry = buffers[buf_idx] if buf_idx < len(buffers) else {"offset": 0, "size": 0}
                    t_info = {
                        "index": ti,
                        "name": t_name,
                        "shape": shape,
                        "type_id": t_type_id,
                        "type": TENSOR_TYPES.get(t_type_id, f"TYPE_{t_type_id}"),
                        "buffer_index": buf_idx,
                        "buffer_offset": buf_entry["offset"],
                        "buffer_size": buf_entry["size"],
                    }
                    tensors.append(t_info)

                    if buf_entry["size"] >= 0xC8 and (
                        "BINARY" in t_name.upper()
                        or data[buf_entry["offset"] : buf_entry["offset"] + 4] == b"\xe0\xfe\x0f\x0c"
                    ):
                        ncp = parse_ncp_header(data, buf_entry["offset"], buf_entry["size"])
                        if ncp is not None:
                            ncp["tensor_name"] = t_name
                            ncp["subgraph_index"] = si
                            ncp_sections.append(ncp)

            ops: list[dict[str, Any]] = []
            ops_pos = fb.field_off(sg, sf, 3)
            if ops_pos is not None:
                opvec = fb.deref(ops_pos)
                nop = fb.u32(opvec)
                for oi in range(nop):
                    op = fb.deref(opvec + 4 + 4 * oi)
                    of = fb.vtable(op)
                    oc_idx = fb.u32(fb.field_off(op, of, 0)) if fb.field_off(op, of, 0) else 0
                    op_in = fb.read_i32_vec(fb.field_off(op, of, 1))
                    op_out = fb.read_i32_vec(fb.field_off(op, of, 2))
                    btype = fb.u8(fb.field_off(op, of, 3)) if fb.field_off(op, of, 3) else 0
                    bopt_pos = fb.field_off(op, of, 4)
                    oc_entry = opcodes[oc_idx] if oc_idx < len(opcodes) else {"name": f"IDX_{oc_idx}", "builtin_code": -1}

                    op_record: dict[str, Any] = {
                        "index": oi,
                        "opcode": oc_entry["name"],
                        "builtin_code": oc_entry["builtin_code"],
                        "inputs": op_in,
                        "outputs": op_out,
                        "builtin_options_type": BUILTIN_OPTIONS_TYPES.get(btype, f"OPT_{btype}"),
                    }

                    if btype == 127 and bopt_pos is not None:
                        bopt = fb.deref(bopt_pos)
                        bf = fb.vtable(bopt)
                        fb_off, fb_len = fb.read_u8_vec(fb.field_off(bopt, bf, 0))
                        if fb_len > 0:
                            kv = parse_flexbuffer_map(data[fb_off : fb_off + fb_len])
                            if kv:
                                compiler_metadata.update(kv)
                                op_record["npu_options_metadata"] = kv
                        bin_name = fb.read_str(fb.field_off(bopt, bf, 1))
                        if bin_name:
                            op_record["npu_binary_name"] = bin_name
                            compiler_metadata.setdefault("BINARY_NAME", bin_name)

                    ops.append(op_record)

            subgraphs.append(
                {
                    "index": si,
                    "name": sg_name,
                    "inputs": [tensors[idx] for idx in in_indices if 0 <= idx < len(tensors)],
                    "outputs": [tensors[idx] for idx in out_indices if 0 <= idx < len(tensors)],
                    "tensors": tensors,
                    "operators": ops,
                }
            )

    # Also detect any raw NCP buffers not directly linked to a tensor name
    known_offsets = {n["file_offset"] for n in ncp_sections}
    for b in buffers:
        if b["size"] >= 0xC8 and b["offset"] not in known_offsets:
            ncp = parse_ncp_header(data, b["offset"], b["size"])
            if ncp is not None:
                ncp["tensor_name"] = f"<buffer_{b['index']}>"
                ncp["subgraph_index"] = None
                ncp_sections.append(ncp)

    # TFLite root metadata table entries (field 6)
    root_metadata: list[dict[str, Any]] = []
    m_pos = fb.field_off(root, rfields, 6)
    if m_pos is not None:
        mvec = fb.deref(m_pos)
        for i in range(fb.u32(mvec)):
            mt = fb.deref(mvec + 4 + 4 * i)
            mf = fb.vtable(mt)
            mname = fb.read_str(fb.field_off(mt, mf, 0))
            mbuf = fb.u32(fb.field_off(mt, mf, 1)) if fb.field_off(mt, mf, 1) else 0
            bsize = buffers[mbuf]["size"] if mbuf < len(buffers) else 0
            root_metadata.append({"name": mname, "buffer_index": mbuf, "size": bsize})

    return {
        "file": str(path),
        "file_size": len(data),
        "magic": magic.decode("ascii"),
        "root_offset": root,
        "schema_version": version,
        "description": description,
        "operator_codes": opcodes,
        "buffers": buffers,
        "compiler_metadata": compiler_metadata,
        "subgraphs": subgraphs,
        "ncp_sections": ncp_sections,
        "root_metadata": root_metadata,
    }


def format_size(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n:,} B ({n / (1024 * 1024):.2f} MiB)"
    if n >= 1024:
        return f"{n:,} B ({n / 1024:.1f} KiB)"
    return f"{n} B"


def print_report(info: dict[str, Any], ansi: Ansi) -> None:
    print(ansi.bold(ansi.cyan(f"=== {Path(info['file']).name} ===")))
    print(
        f"  {ansi.bold('Container')}: magic={ansi.green(info['magic'])}  "
        f"size={format_size(info['file_size'])}  "
        f"root_off=0x{info['root_offset']:x}  schema_ver={info['schema_version']}"
    )

    meta = info["compiler_metadata"]
    if meta:
        print(ansi.bold("  Compiler / ENN_NPUOptions Metadata:"))
        preferred = [
            "SOC_TYPE",
            "QUANT_DEV",
            "NPUC_VERSION",
            "NCP_VERSION",
            "FRAMEWORK",
            "QUANT_MODE",
            "QUANT_BW",
            "HW_CFU",
            "OPT_LEVEL",
            "NAME",
            "ONNX",
            "MODEL",
            "PROTOTXT",
            "PROTOBIN",
            "COMPILED_COMMAND",
        ]
        seen = set()
        for key in preferred + [k for k in meta if k not in preferred]:
            if key in meta and key not in seen:
                seen.add(key)
                val = str(meta[key]).strip()
                if val == "":
                    val = ansi.dim("<empty>")
                print(f"    {ansi.yellow(f'{key:18s}')}: {val}")

    for sg in info["subgraphs"]:
        print(ansi.bold(f"  Subgraph[{sg['index']}]: ") + ansi.magenta(sg["name"]))
        print(f"    Inputs ({len(sg['inputs'])}):")
        for t in sg["inputs"]:
            print(f"      [{t['index']}] {t['name']}  shape={t['shape']}  type={ansi.green(t['type'])}")
        print(f"    Outputs ({len(sg['outputs'])}):")
        for t in sg["outputs"]:
            print(f"      [{t['index']}] {t['name']}  shape={t['shape']}  type={ansi.green(t['type'])}")
        op_summary = ", ".join(f"{op['opcode']}({op['builtin_code']})" for op in sg["operators"])
        print(f"    Operators ({len(sg['operators'])}): {op_summary}")
        npu_bins = [t for t in sg["tensors"] if t["buffer_size"] > 0]
        for t in npu_bins:
            print(
                f"    Embedded Tensor[{t['index']}] {ansi.cyan(t['name'])}: "
                f"offset=0x{t['buffer_offset']:x}  size={format_size(t['buffer_size'])}  shape={t['shape']}  type={t['type']}"
            )

    if info["ncp_sections"]:
        for ncp in info["ncp_sections"]:
            m2_tag = ansi.green("OK") if ncp["magic2_valid"] else ansi.yellow("MISMATCH")
            print(ansi.bold("  NCP Binary Header:"))
            print(
                f"    tensor={ncp['tensor_name']}  offset=0x{ncp['file_offset']:x}  "
                f"buf_size={format_size(ncp['buffer_size'])}"
            )
            print(
                f"    magic1={ncp['magic1']}  magic2={ncp['magic2']} ({m2_tag})  "
                f"ncp_ver={ncp['hdr_version']}  hdr_size={ncp['hdr_size']} B  "
                f"model_name={ncp['model_name']!r}"
            )
            print(
                f"    body_ver=0x{ncp['body_version']:x}  body_off=+0x{ncp['body_offset']:x}  "
                f"body_size={format_size(ncp['body_size'])}  "
                f"kops={ncp['computational_workload_kops']:,}  io_kb={ncp['io_workload_kb']:,}"
            )
            print(
                f"    vectors: addr={ncp['address_vector_cnt']}  mem={ncp['memory_vector_cnt']}  "
                f"grp={ncp['group_vector_cnt']}  thr={ncp['thread_vector_cnt']}  "
                f"io={ncp['io_vector_cnt']}  llc={ncp['llc_vector_cnt']}"
            )
    else:
        print(ansi.yellow("  NCP Binary Header: none found"))
    print()


def collect_paths(targets: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in targets:
        p = Path(raw)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.nnc")))
        else:
            paths.append(p)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect Samsung ENN (.nnc / ENNC FlatBuffer) model containers."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Path(s) to .nnc files or directories containing .nnc files.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output parsed container metadata as JSON.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color codes in CLI output.",
    )
    args = parser.parse_args(argv)

    targets = collect_paths(args.paths)
    if not targets:
        print("No .nnc files found.", file=sys.stderr)
        return 1

    results: list[dict[str, Any]] = []
    errors = 0
    for target in targets:
        try:
            results.append(parse_nnc(target))
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"Error parsing {target}: {exc}", file=sys.stderr)

    if args.json:
        payload: Any = results[0] if len(results) == 1 else results
        print(json.dumps(payload, indent=2))
    else:
        use_color = (not args.no_color) and (
            sys.stdout.isatty() or os.environ.get("FORCE_COLOR") == "1"
        )
        # Enable ANSI sequences cleanly on Windows terminals when supported
        ansi = Ansi(enabled=not args.no_color and (use_color or os.name != "nt" or "WT_SESSION" in os.environ or "TERM" in os.environ))
        for item in results:
            print_report(item, ansi)

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
