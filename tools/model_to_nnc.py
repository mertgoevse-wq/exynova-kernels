#!/usr/bin/env python3
"""Translate and validate TFLite (.tflite / TFL3) models into Samsung ENN (.nnc / ENNC) containers.

Stdlib-only Python 3 tool for ExyNova:
  - tflite2nnc <input.tflite> <output.nnc>
      Validates FlatBuffer structure, rewrites identifier 'TFL3' -> 'ENNC',
      prints an operator-level compatibility report (CPU / Xclipse 540 Vulkan /
      Samsung ENN Unified vs ENN_NPU NCP v26), and writes the .nnc container.
  - build-add-nnc <output.nnc>
      Synthesizes a minimal valid ENNC FlatBuffer ADD model from scratch
      (TFLite v3 schema layout) for offline testing without needing a .tflite input.
  - build-add-tflite <output.tflite>
      Synthesizes a minimal valid TFL3 FlatBuffer ADD model for round-trip tests.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path
from typing import Any

# Standard TFLite BuiltinOperator IDs + Samsung ENN extensions (1012/1013)
TFLITE_BUILTIN_OPS: dict[int, str] = {
    0: "ADD",
    1: "AVERAGE_POOL_2D",
    2: "CONCATENATION",
    3: "CONV_2D",
    4: "DEPTHWISE_CONV_2D",
    5: "DEPTH_TO_SPACE",
    6: "DEQUANTIZE",
    7: "EMBEDDING_LOOKUP",
    8: "FLOOR",
    9: "FULLY_CONNECTED",
    10: "HASHTABLE_LOOKUP",
    11: "L2_NORMALIZATION",
    12: "L2_POOL_2D",
    13: "LOCAL_RESPONSE_NORMALIZATION",
    14: "LOGISTIC",
    15: "LSH_PROJECTION",
    16: "LSTM",
    17: "MAX_POOL_2D",
    18: "MUL",
    19: "RELU",
    20: "RELU_N1_TO_1",
    21: "RELU6",
    22: "RESHAPE",
    23: "RESIZE_BILINEAR",
    24: "RNN",
    25: "SOFTMAX",
    26: "SPACE_TO_DEPTH",
    27: "SVDF",
    28: "TANH",
    29: "CONCAT_EMBEDDINGS",
    30: "SKIP_GRAM",
    31: "CALL",
    32: "CUSTOM",
    33: "EMBEDDING_LOOKUP_SPARSE",
    34: "PAD",
    35: "UNIDIRECTIONAL_SEQUENCE_RNN",
    36: "GATHER",
    37: "BATCH_TO_SPACE_ND",
    38: "SPACE_TO_BATCH_ND",
    39: "TRANSPOSE",
    40: "MEAN",
    41: "SUB",
    42: "DIV",
    43: "SQUEEZE",
    44: "UNIDIRECTIONAL_SEQUENCE_LSTM",
    45: "STRIDED_SLICE",
    46: "BIDIRECTIONAL_SEQUENCE_RNN",
    47: "EXP",
    48: "TOPK_V2",
    49: "SPLIT",
    50: "LOG_SOFTMAX",
    53: "CAST",
    54: "PRELU",
    55: "MAXIMUM",
    56: "ARG_MAX",
    57: "MINIMUM",
    58: "LESS",
    59: "NEG",
    60: "PADV2",
    61: "GREATER",
    62: "GREATER_EQUAL",
    63: "LESS_EQUAL",
    64: "SELECT",
    65: "SLICE",
    66: "SIN",
    67: "TRANSPOSE_CONV",
    71: "EXPAND_DIMS",
    72: "PACK",
    73: "REDUCE_MAX",
    74: "RSQRT",
    75: "SHAPE",
    76: "POW",
    77: "ARG_MIN",
    78: "FAKE_QUANT",
    81: "REDUCE_PROD",
    82: "REDUCE_MIN",
    83: "STRIDED_SLICE",
    84: "GELU",
    89: "SQRT",
    90: "LOG",
    91: "ABS",
    97: "RESIZE_NEAREST_NEIGHBOR",
    98: "LEAKY_RELU",
    102: "HARD_SWISH",
    114: "QUANTIZE",
    127: "SILU",
    1012: "ENN_PREPROC_OR_CUSTOM_1012",
    1013: "ENN_NPU",
}

TENSOR_TYPES: dict[int, str] = {
    0: "FLOAT32",
    1: "FLOAT16",
    2: "INT32",
    3: "UINT8",
    4: "INT64",
    5: "STRING",
    6: "BOOL",
    7: "INT16",
    9: "INT8",
    10: "FLOAT64",
    16: "UINT16",
    17: "INT4",
    18: "BFLOAT16",
}

# Ops natively mapped to ExyNova's verified Xclipse 540 Vulkan compute kernels
VULKAN_NATIVE_OPS = {
    "ADD",
    "MUL",
    "FULLY_CONNECTED",
    "CONV_2D",
    "RELU",
    "SILU",
    "SOFTMAX",
}

# Ops supported by ExyNova CPU reference/NEON or trivial shape metadata ops
CPU_SUPPORTED_OPS = VULKAN_NATIVE_OPS | {
    "RESHAPE",
    "SQUEEZE",
    "TRANSPOSE",
    "QUANTIZE",
    "DEQUANTIZE",
    "SUB",
    "CONCATENATION",
    "AVERAGE_POOL_2D",
    "MAX_POOL_2D",
    "PAD",
}


class FlatBufferValidator:
    """Bounds-checked reader and validator for TFLite v3 / ENNC FlatBuffers."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.size = len(data)

    def _check(self, off: int, length: int, label: str) -> None:
        if off < 0 or length < 0 or off + length > self.size:
            raise ValueError(
                f"Out-of-bounds read ({label}): offset=0x{off:x}, len={length}, file_size={self.size}"
            )

    def u8(self, off: int) -> int:
        self._check(off, 1, "u8")
        return self.data[off]

    def i8(self, off: int) -> int:
        self._check(off, 1, "i8")
        return struct.unpack_from("<b", self.data, off)[0]

    def u16(self, off: int) -> int:
        self._check(off, 2, "u16")
        return struct.unpack_from("<H", self.data, off)[0]

    def u32(self, off: int) -> int:
        self._check(off, 4, "u32")
        return struct.unpack_from("<I", self.data, off)[0]

    def i32(self, off: int) -> int:
        self._check(off, 4, "i32")
        return struct.unpack_from("<i", self.data, off)[0]

    def vtable(self, table_off: int) -> list[int]:
        vt_rel = self.i32(table_off)
        vt_off = table_off - vt_rel
        vsize = self.u16(vt_off)
        tsize = self.u16(vt_off + 2)
        if vsize < 4 or vsize % 2 != 0:
            raise ValueError(f"Invalid vtable size {vsize} at 0x{vt_off:x}")
        if tsize < 4:
            raise ValueError(f"Invalid table size {tsize} at 0x{table_off:x}")
        self._check(vt_off, vsize, "vtable")
        self._check(table_off, tsize, "table")
        count = (vsize - 4) // 2
        fields = [self.u16(vt_off + 4 + 2 * i) for i in range(count)]
        for f in fields:
            if f != 0 and f >= tsize:
                raise ValueError(f"vtable field offset {f} >= table size {tsize}")
        return fields

    def field_off(self, table_off: int, fields: list[int], idx: int) -> int | None:
        if idx >= len(fields) or fields[idx] == 0:
            return None
        return table_off + fields[idx]

    def deref(self, off: int) -> int:
        rel = self.u32(off)
        target = off + rel
        self._check(target, 4, "deref")
        return target

    def read_str(self, field_pos: int | None) -> str | None:
        if field_pos is None:
            return None
        s_off = self.deref(field_pos)
        length = self.u32(s_off)
        self._check(s_off + 4, length, "string")
        return self.data[s_off + 4 : s_off + 4 + length].decode("utf-8", "replace")

    def read_i32_vec(self, field_pos: int | None) -> list[int]:
        if field_pos is None:
            return []
        v_off = self.deref(field_pos)
        count = self.u32(v_off)
        self._check(v_off + 4, 4 * count, "i32_vec")
        return [self.i32(v_off + 4 + 4 * i) for i in range(count)]


def validate_and_inspect_model(data: bytes) -> dict[str, Any]:
    """Validate FlatBuffer structure and extract TFLite/ENNC operator & tensor info."""
    if len(data) < 12:
        raise ValueError(f"File too small ({len(data)} bytes); minimum FlatBuffer header is 12 bytes")

    magic = data[4:8]
    if magic not in (b"TFL3", b"ENNC"):
        raise ValueError(f"Unsupported FlatBuffer identifier {magic!r}; expected b'TFL3' or b'ENNC'")

    fb = FlatBufferValidator(data)
    root = fb.u32(0)
    if root < 8 or root + 4 > len(data):
        raise ValueError(f"Root offset 0x{root:x} out of bounds for size {len(data)}")
    rfields = fb.vtable(root)

    version_pos = fb.field_off(root, rfields, 0)
    version = fb.u32(version_pos) if version_pos is not None else 0
    description = fb.read_str(fb.field_off(root, rfields, 3)) or ""

    # Validate buffers vector
    num_buffers = 0
    bvec_pos = fb.field_off(root, rfields, 4)
    if bvec_pos is not None:
        bvec = fb.deref(bvec_pos)
        num_buffers = fb.u32(bvec)
        fb._check(bvec + 4, 4 * num_buffers, "buffers_vec")
        for i in range(num_buffers):
            bt = fb.deref(bvec + 4 + 4 * i)
            fb.vtable(bt)

    # Validate operator_codes
    opcodes: list[dict[str, Any]] = []
    oc_pos = fb.field_off(root, rfields, 1)
    if oc_pos is not None:
        ocvec = fb.deref(oc_pos)
        noc = fb.u32(ocvec)
        fb._check(ocvec + 4, 4 * noc, "opcodes_vec")
        for i in range(noc):
            ot = fb.deref(ocvec + 4 + 4 * i)
            of = fb.vtable(ot)
            dep_pos = fb.field_off(ot, of, 0)
            dep_code = fb.i8(dep_pos) if dep_pos is not None else 0
            custom = fb.read_str(fb.field_off(ot, of, 1))
            ver_pos = fb.field_off(ot, of, 2)
            op_ver = fb.i32(ver_pos) if ver_pos is not None else 1
            bcode_pos = fb.field_off(ot, of, 3)
            bcode = fb.i32(bcode_pos) if bcode_pos is not None else dep_code
            if bcode == 0 and dep_code not in (0, 127):
                bcode = dep_code
            name = custom if custom else TFLITE_BUILTIN_OPS.get(bcode, f"OP_{bcode}")
            opcodes.append(
                {
                    "index": i,
                    "builtin_code": bcode,
                    "custom_code": custom,
                    "version": op_ver,
                    "name": name,
                }
            )

    # Validate subgraphs, tensors, and operators
    subgraphs: list[dict[str, Any]] = []
    sg_pos = fb.field_off(root, rfields, 2)
    if sg_pos is not None:
        sgvec = fb.deref(sg_pos)
        nsg = fb.u32(sgvec)
        fb._check(sgvec + 4, 4 * nsg, "subgraphs_vec")
        for si in range(nsg):
            sg = fb.deref(sgvec + 4 + 4 * si)
            sf = fb.vtable(sg)
            sg_name = fb.read_str(fb.field_off(sg, sf, 4)) or f"subgraph_{si}"
            sg_inputs = fb.read_i32_vec(fb.field_off(sg, sf, 1))
            sg_outputs = fb.read_i32_vec(fb.field_off(sg, sf, 2))

            tensors: list[dict[str, Any]] = []
            t_pos = fb.field_off(sg, sf, 0)
            if t_pos is not None:
                tvec = fb.deref(t_pos)
                nt = fb.u32(tvec)
                fb._check(tvec + 4, 4 * nt, "tensors_vec")
                for ti in range(nt):
                    tt = fb.deref(tvec + 4 + 4 * ti)
                    tf = fb.vtable(tt)
                    shape = fb.read_i32_vec(fb.field_off(tt, tf, 0))
                    type_pos = fb.field_off(tt, tf, 1)
                    t_type_id = fb.u8(type_pos) if type_pos is not None else 0
                    buf_pos = fb.field_off(tt, tf, 2)
                    buf_idx = fb.u32(buf_pos) if buf_pos is not None else 0
                    t_name = fb.read_str(fb.field_off(tt, tf, 3)) or f"tensor_{ti}"
                    tensors.append(
                        {
                            "index": ti,
                            "name": t_name,
                            "shape": shape,
                            "type_id": t_type_id,
                            "type": TENSOR_TYPES.get(t_type_id, f"TYPE_{t_type_id}"),
                            "buffer_index": buf_idx,
                        }
                    )

            operators: list[dict[str, Any]] = []
            ops_pos = fb.field_off(sg, sf, 3)
            if ops_pos is not None:
                opvec = fb.deref(ops_pos)
                nop = fb.u32(opvec)
                fb._check(opvec + 4, 4 * nop, "operators_vec")
                for oi in range(nop):
                    op = fb.deref(opvec + 4 + 4 * oi)
                    of = fb.vtable(op)
                    oc_pos_i = fb.field_off(op, of, 0)
                    oc_idx = fb.u32(oc_pos_i) if oc_pos_i is not None else 0
                    if oc_idx >= len(opcodes):
                        raise ValueError(
                            f"Subgraph {si} operator {oi}: opcode_index {oc_idx} >= {len(opcodes)}"
                        )
                    op_in = fb.read_i32_vec(fb.field_off(op, of, 1))
                    op_out = fb.read_i32_vec(fb.field_off(op, of, 2))
                    btype_pos = fb.field_off(op, of, 3)
                    btype = fb.u8(btype_pos) if btype_pos is not None else 0
                    thw_pos = fb.field_off(op, of, 9)
                    target_hw = fb.i8(thw_pos) if thw_pos is not None else None
                    oc = opcodes[oc_idx]
                    operators.append(
                        {
                            "index": oi,
                            "opcode": oc["name"],
                            "builtin_code": oc["builtin_code"],
                            "inputs": op_in,
                            "outputs": op_out,
                            "builtin_options_type": btype,
                            "target_hw": target_hw,
                        }
                    )

            subgraphs.append(
                {
                    "index": si,
                    "name": sg_name,
                    "inputs": sg_inputs,
                    "outputs": sg_outputs,
                    "tensors": tensors,
                    "operators": operators,
                }
            )

    has_ncp = b"\xe0\xfe\x0f\x0c" in data
    return {
        "size": len(data),
        "magic": magic.decode("ascii"),
        "root_offset": root,
        "schema_version": version,
        "description": description,
        "num_buffers": num_buffers,
        "operator_codes": opcodes,
        "subgraphs": subgraphs,
        "has_ncp_binary": has_ncp,
    }


def classify_operator(op_name: str, builtin_code: int, has_ncp: bool) -> dict[str, str]:
    """Return compatibility matrix for an operator on Exynos 1580 / ENN drivers."""
    if builtin_code == 1013 or op_name == "ENN_NPU":
        return {
            "cpu": "No (NPU blob)",
            "gpu_vulkan": "No (NPU blob)",
            "enn_unified": "Yes (App context)",
            "enn_npu_ncp26": "YES (NCP v26 ready)" if has_ncp else "Missing NCP binary",
        }
    if builtin_code == 1012:
        return {
            "cpu": "Custom preproc",
            "gpu_vulkan": "Shader/Host",
            "enn_unified": "Yes (ENN custom)",
            "enn_npu_ncp26": "Pre-NPU stage",
        }
    cpu_ok = "YES" if op_name in CPU_SUPPORTED_OPS else "Fallback/Ext"
    gpu_ok = "YES (Xclipse SPV)" if op_name in VULKAN_NATIVE_OPS else "CPU fallback"
    return {
        "cpu": cpu_ok,
        "gpu_vulkan": gpu_ok,
        "enn_unified": "Unified CPU/GPU only",
        "enn_npu_ncp26": "NO (needs NPUC/NCP v26)",
    }


def print_compatibility_report(info: dict[str, Any], out_magic: str) -> None:
    """Print operator-level compatibility report for Samsung ENN vs ExyNova backends."""
    print(f"=== ExyNova Model Translator ({info['magic']} -> {out_magic}) ===")
    print(
        f"  Size: {info['size']:,} B | root_off=0x{info['root_offset']:x} | "
        f"schema_ver={info['schema_version']} | buffers={info['num_buffers']} | "
        f"ncp_v26_embedded={'YES' if info['has_ncp_binary'] else 'NO'}"
    )
    if info["description"]:
        print(f"  Description: {info['description']}")

    for sg in info["subgraphs"]:
        print(
            f"\n  Subgraph[{sg['index']}] '{sg['name']}': "
            f"tensors={len(sg['tensors'])}, inputs={sg['inputs']}, outputs={sg['outputs']}"
        )
        if not sg["operators"]:
            print("    (no operators)")
            continue
        print(
            f"    {'#':<3} {'Operator':<24} {'Code':<6} {'CPU':<12} "
            f"{'Vulkan GPU':<18} {'ENN Unified':<21} {'ENN_NPU (NCP v26)'}"
        )
        print("    " + "-" * 108)
        for op in sg["operators"]:
            c = classify_operator(op["opcode"], op["builtin_code"], info["has_ncp_binary"])
            print(
                f"    {op['index']:<3} {op['opcode']:<24} {op['builtin_code']:<6} "
                f"{c['cpu']:<12} {c['gpu_vulkan']:<18} {c['enn_unified']:<21} {c['enn_npu_ncp26']}"
            )
    print(
        "\n  Note: Rewriting 'TFL3' -> 'ENNC' produces a valid ENNC FlatBuffer container for\n"
        "  ENN Unified CPU/GPU paths and nnc_inspect.py. Direct Samsung NPU execution\n"
        "  (NpuUserDriver) additionally requires offline NPUC/nncontainer-gen lowering\n"
        "  into an ENN_NPU (1013) op + NCP v26 binary (0x0C0FFEE0 / 0xC0DEC0DE)."
    )


class MiniFlatBufferBuilder:
    """Minimal backward-growing FlatBuffer builder (TFLite v3 / ENNC compatible)."""

    def __init__(self, initial_size: int = 8192) -> None:
        self.buf = bytearray(initial_size)
        self.head = initial_size

    def _ensure(self, extra: int) -> None:
        if self.head >= extra:
            return
        grow = max(len(self.buf), extra + 4096)
        self.buf = bytearray(grow) + self.buf
        self.head += grow

    def _align(self, align: int) -> None:
        cur = len(self.buf) - self.head
        rem = cur % align
        if rem != 0:
            pad = align - rem
            self._ensure(pad)
            self.head -= pad

    def offset(self) -> int:
        return len(self.buf) - self.head

    def create_string(self, text: str) -> int:
        raw = text.encode("utf-8")
        self._align(4)
        total = (4 + len(raw) + 1 + 3) & ~3
        self._ensure(total)
        self.head -= total
        struct.pack_into("<I", self.buf, self.head, len(raw))
        self.buf[self.head + 4 : self.head + 4 + len(raw)] = raw
        return self.offset()

    def create_i32_vector(self, values: list[int]) -> int:
        self._align(4)
        total = 4 + 4 * len(values)
        self._ensure(total)
        self.head -= total
        struct.pack_into("<I", self.buf, self.head, len(values))
        for i, v in enumerate(values):
            struct.pack_into("<i", self.buf, self.head + 4 + 4 * i, v)
        return self.offset()

    def create_u8_vector(self, raw: bytes) -> int:
        self._align(4)
        total = 4 + ((len(raw) + 3) & ~3)
        self._ensure(total)
        self.head -= total
        struct.pack_into("<I", self.buf, self.head, len(raw))
        if raw:
            self.buf[self.head + 4 : self.head + 4 + len(raw)] = raw
        return self.offset()

    def create_offset_vector(self, target_offsets: list[int]) -> int:
        self._align(4)
        total = 4 + 4 * len(target_offsets)
        self._ensure(total)
        self.head -= total
        struct.pack_into("<I", self.buf, self.head, len(target_offsets))
        for i, target_off in enumerate(target_offsets):
            elem_pos = self.head + 4 + 4 * i
            elem_off = len(self.buf) - elem_pos
            struct.pack_into("<I", self.buf, elem_pos, elem_off - target_off)
        return self.offset()

    def create_table(self, num_fields: int, fields: dict[int, tuple[str, int]]) -> int:
        # Layout fields: 4-byte scalars/offsets first, then 1-byte scalars
        rel_offsets: dict[int, int] = {}
        cur = 4  # reserve 4 bytes for vtable soffset
        order_4 = [k for k, (kind, _) in sorted(fields.items()) if kind in ("u32", "i32", "off")]
        order_1 = [k for k, (kind, _) in sorted(fields.items()) if kind in ("u8", "i8")]
        for k in order_4:
            rel_offsets[k] = cur
            cur += 4
        for k in order_1:
            rel_offsets[k] = cur
            cur += 1
        table_size = (cur + 3) & ~3

        self._align(4)
        self._ensure(table_size)
        self.head -= table_size
        table_pos = self.head
        table_off = len(self.buf) - table_pos

        for k, (kind, val) in fields.items():
            f_pos = table_pos + rel_offsets[k]
            if kind == "u32":
                struct.pack_into("<I", self.buf, f_pos, val)
            elif kind == "i32":
                struct.pack_into("<i", self.buf, f_pos, val)
            elif kind == "u8":
                struct.pack_into("<B", self.buf, f_pos, val)
            elif kind == "i8":
                struct.pack_into("<b", self.buf, f_pos, val)
            elif kind == "off":
                f_off = len(self.buf) - f_pos
                struct.pack_into("<I", self.buf, f_pos, f_off - val)

        vsize = 4 + 2 * num_fields
        vsize_aligned = (vsize + 3) & ~3
        self._align(4)
        self._ensure(vsize_aligned)
        self.head -= vsize_aligned
        vt_pos = self.head
        vt_off = len(self.buf) - vt_pos
        struct.pack_into("<HH", self.buf, vt_pos, vsize, table_size)
        for idx in range(num_fields):
            struct.pack_into("<H", self.buf, vt_pos + 4 + 2 * idx, rel_offsets.get(idx, 0))

        # Write positive soffset from table_pos back to vt_pos (vt_pos is earlier in final buffer -> Wait!)
        # Wait: in backward builder, vt_pos < table_pos in self.buf!
        # In final buffer, vt is at (vt_pos - head) and table is at (table_pos - head), so vt < table!
        # table_file_off - i32(table_file_off) == vt_file_off  =>  i32 = table_file_off - vt_file_off = vt_off - table_off > 0!
        struct.pack_into("<i", self.buf, table_pos, vt_off - table_off)
        return table_off

    def finish(self, root_table_off: int, identifier: bytes = b"ENNC") -> bytes:
        if len(identifier) != 4:
            raise ValueError("FlatBuffer file identifier must be 4 bytes")
        self._align(4)
        self._ensure(8)
        self.head -= 8
        header_off = len(self.buf) - self.head
        struct.pack_into("<I", self.buf, self.head, header_off - root_table_off)
        self.buf[self.head + 4 : self.head + 8] = identifier
        return bytes(self.buf[self.head :])


def build_minimal_add_flatbuffer(
    identifier: bytes = b"ENNC",
    schema_version: int = 211,
    target_hw: int | None = 2,
) -> bytes:
    """Synthesize a valid TFLite / ENNC v211 FlatBuffer ADD model matching A56 NncParseStrategy."""
    b = MiniFlatBufferBuilder()

    # Buffers: 0 (sentinel), 1 (x), 2 (y), 3 (output)
    buf_offs = [b.create_table(1, {}) for _ in range(4)]
    buffers_vec = b.create_offset_vector(buf_offs)

    # OperatorCode[0]: ADD (builtin_code = 0, version = 1)
    op_code_0 = b.create_table(
        4,
        {
            0: ("i8", 0),   # deprecated_builtin_code = ADD(0)
            2: ("i32", 1),  # version = 1
            3: ("i32", 0),  # builtin_code = ADD(0)
        },
    )
    op_codes_vec = b.create_offset_vector([op_code_0])

    # Tensors: 4D NCHW [1, 1, 1, 4] + ENNC v211 extension fields (prev/next operators, s5=1, s11=100)
    empty_ops = b.create_i32_vector([])
    next_op0 = b.create_i32_vector([0])
    tensor_specs = [
        ("input_x", [1, 1, 1, 4], 1, None, next_op0),
        ("input_y", [1, 1, 1, 4], 2, None, next_op0),
        ("output_add", [1, 1, 1, 4], 3, next_op0, empty_ops),
    ]
    tensor_offs = []
    for name, shape, buf_idx, prev_v, next_v in tensor_specs:
        shape_vec = b.create_i32_vector(shape)
        name_str = b.create_string(name)
        tfields: dict[int, tuple[str, int]] = {
            0: ("off", shape_vec),  # shape
            1: ("u8", 0),           # type = FLOAT32 (0)
            2: ("u32", buf_idx),    # buffer
            3: ("off", name_str),   # name
        }
        if target_hw is not None:
            tfields[5] = ("u8", 1)
            if prev_v is not None:
                tfields[8] = ("off", prev_v)
            if next_v is not None:
                tfields[9] = ("off", next_v)
            tfields[11] = ("u8", 100)
        tensor_offs.append(b.create_table(12 if target_hw is not None else 4, tfields))
    tensors_vec = b.create_offset_vector(tensor_offs)

    # AddOptions table (builtin_options_type = 1 = AddOptions, slot 0 = fused_activation_function NONE(0))
    add_opts = b.create_table(2, {0: ("i8", 0)})

    # Operator[0]: ADD(inputs=[0, 1], outputs=[2]) + builtin_options + ENN TargetHw slot 9
    op_in_vec = b.create_i32_vector([0, 1])
    op_out_vec = b.create_i32_vector([2])
    op_fields: dict[int, tuple[str, int]] = {
        0: ("u32", 0),        # opcode_index
        1: ("off", op_in_vec),
        2: ("off", op_out_vec),
        3: ("u8", 1),         # builtin_options_type = AddOptions (1)
        4: ("off", add_opts), # mandatory non-null builtin_options table
    }
    if target_hw is not None:
        op_fields[9] = ("i8", int(target_hw))
    op_0 = b.create_table(10 if target_hw is not None else 5, op_fields)
    ops_vec = b.create_offset_vector([op_0])

    # SubGraph[0]
    sg_in_vec = b.create_i32_vector([0, 1])
    sg_out_vec = b.create_i32_vector([2])
    sg_name = b.create_string("main")
    sg_0 = b.create_table(
        5,
        {
            0: ("off", tensors_vec),
            1: ("off", sg_in_vec),
            2: ("off", sg_out_vec),
            3: ("off", ops_vec),
            4: ("off", sg_name),
        },
    )
    sgs_vec = b.create_offset_vector([sg_0])

    desc_str = b.create_string(f"ExyNova ENNC v{schema_version} ADD TargetHw={target_hw}")
    model_root = b.create_table(
        7,
        {
            0: ("u32", schema_version),
            1: ("off", op_codes_vec),
            2: ("off", sgs_vec),
            3: ("off", desc_str),
            4: ("off", buffers_vec),
        },
    )
    return b.finish(model_root, identifier=identifier)


def build_mlp_ennc_flatbuffer(
    dim: int = 16,
    target_hw: int = 2,
    schema_version: int = 211,
) -> bytes:
    """Synthesize a multi-layer MLP ENNC FlatBuffer with Operator.TargetHw (slot 9)."""
    b = MiniFlatBufferBuilder()

    # Constant FP32 weights: W1[dim, dim], B1[dim], W2[dim, dim], B2[dim]
    w1_bytes = struct.pack(f"<{dim * dim}f", *([0.05] * (dim * dim)))
    b1_bytes = struct.pack(f"<{dim}f", *([0.01] * dim))
    w2_bytes = struct.pack(f"<{dim * dim}f", *([0.04] * (dim * dim)))
    b2_bytes = struct.pack(f"<{dim}f", *([0.0] * dim))

    buf_offs = [b.create_table(1, {}) for _ in range(5)]  # 0=sentinel,1=in,2=h1,3=h2,4=out
    for raw in (w1_bytes, b1_bytes, w2_bytes, b2_bytes):  # buffers 5..8
        vec_off = b.create_u8_vector(raw)
        buf_offs.append(b.create_table(1, {0: ("off", vec_off)}))
    buffers_vec = b.create_offset_vector(buf_offs)

    # OperatorCodes: FULLY_CONNECTED(9), RELU(19), SOFTMAX(25)
    oc_specs = [(9, 1), (19, 1), (25, 1)]
    oc_offs = [
        b.create_table(4, {0: ("i8", code), 2: ("i32", ver), 3: ("i32", code)})
        for code, ver in oc_specs
    ]
    op_codes_vec = b.create_offset_vector(oc_offs)

    tensor_defs = [
        ("x_in", [1, dim], 1),       # 0
        ("w1", [dim, dim], 5),       # 1
        ("b1", [dim], 6),            # 2
        ("h1_fc", [1, dim], 2),      # 3
        ("h1_relu", [1, dim], 3),    # 4
        ("w2", [dim, dim], 7),       # 5
        ("b2", [dim], 8),            # 6
        ("logits", [1, dim], 4),     # 7
    ]
    tensor_offs = []
    for name, shape, buf_idx in tensor_defs:
        svec = b.create_i32_vector(shape)
        nstr = b.create_string(name)
        tensor_offs.append(
            b.create_table(4, {0: ("off", svec), 1: ("u8", 0), 2: ("u32", buf_idx), 3: ("off", nstr)})
        )
    tensors_vec = b.create_offset_vector(tensor_offs)

    def make_op(oc_idx: int, ins: list[int], outs: list[int]) -> int:
        inv = b.create_i32_vector(ins)
        outv = b.create_i32_vector(outs)
        return b.create_table(
            10,
            {
                0: ("u32", oc_idx),
                1: ("off", inv),
                2: ("off", outv),
                3: ("u8", 0),
                9: ("i8", int(target_hw)),
            },
        )

    ops_vec = b.create_offset_vector(
        [
            make_op(0, [0, 1, 2], [3]),  # FULLY_CONNECTED
            make_op(1, [3], [4]),        # RELU
            make_op(0, [4, 5, 6], [7]),  # FULLY_CONNECTED
        ]
    )

    sg_in_vec = b.create_i32_vector([0])
    sg_out_vec = b.create_i32_vector([7])
    sg_name = b.create_string("main")
    sg_0 = b.create_table(
        5,
        {
            0: ("off", tensors_vec),
            1: ("off", sg_in_vec),
            2: ("off", sg_out_vec),
            3: ("off", ops_vec),
            4: ("off", sg_name),
        },
    )
    sgs_vec = b.create_offset_vector([sg_0])
    desc_str = b.create_string(f"ExyNova MLP ENNC v{schema_version} TargetHw={target_hw}")
    root = b.create_table(
        7,
        {
            0: ("u32", schema_version),
            1: ("off", op_codes_vec),
            2: ("off", sgs_vec),
            3: ("off", desc_str),
            4: ("off", buffers_vec),
        },
    )
    return b.finish(root, identifier=b"ENNC")


def convert_tflite_to_nnc(input_path: Path, output_path: Path, quiet: bool = False) -> dict[str, Any]:
    """Validate input .tflite/.nnc FlatBuffer, rewrite 'TFL3' -> 'ENNC', and write output."""
    raw = input_path.read_bytes()
    info = validate_and_inspect_model(raw)

    out_bytes = bytearray(raw)
    out_bytes[4:8] = b"ENNC"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bytes(out_bytes))

    if not quiet:
        print_compatibility_report(info, "ENNC")
        print(f"\n  Wrote ENNC container: {output_path} ({len(out_bytes):,} bytes)")
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Translate TFLite (.tflite / TFL3) models into Samsung ENN (.nnc / ENNC) containers."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_conv = sub.add_parser(
        "tflite2nnc",
        help="Validate a .tflite FlatBuffer, rewrite 'TFL3' -> 'ENNC', and print ENN compatibility report.",
    )
    p_conv.add_argument("input", type=Path, help="Input .tflite (or .nnc) model path.")
    p_conv.add_argument("output", type=Path, help="Output .nnc file path.")
    p_conv.add_argument("--quiet", action="store_true", help="Suppress compatibility report.")

    p_add_nnc = sub.add_parser(
        "build-add-nnc",
        help="Synthesize a minimal valid ENNC FlatBuffer ADD model from scratch.",
    )
    p_add_nnc.add_argument("output", type=Path, help="Output .nnc file path.")
    p_add_nnc.add_argument("--target-hw", type=int, default=2, choices=[0, 1, 2, 3])
    p_add_nnc.add_argument("--schema-version", type=int, default=211)

    p_mlp_nnc = sub.add_parser(
        "build-mlp-nnc",
        help="Synthesize a multi-layer MLP ENNC FlatBuffer with Operator.TargetHw (slot 9).",
    )
    p_mlp_nnc.add_argument("output", type=Path, help="Output .nnc file path.")
    p_mlp_nnc.add_argument("--dim", type=int, default=16)
    p_mlp_nnc.add_argument("--target-hw", type=int, default=2, choices=[0, 1, 2, 3])
    p_mlp_nnc.add_argument("--schema-version", type=int, default=211)

    p_add_tfl = sub.add_parser(
        "build-add-tflite",
        help="Synthesize a minimal valid TFL3 FlatBuffer ADD model from scratch.",
    )
    p_add_tfl.add_argument("output", type=Path, help="Output .tflite file path.")

    args = parser.parse_args(argv)

    if args.command == "tflite2nnc":
        if not args.input.is_file():
            print(f"Error: input file not found: {args.input}", file=sys.stderr)
            return 1
        try:
            convert_tflite_to_nnc(args.input, args.output, quiet=args.quiet)
        except Exception as exc:  # noqa: BLE001
            print(f"Error converting {args.input}: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "build-add-nnc":
        data = build_minimal_add_flatbuffer(
            b"ENNC", schema_version=args.schema_version, target_hw=args.target_hw
        )
        info = validate_and_inspect_model(data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(data)
        print_compatibility_report(info, "ENNC")
        print(f"\n  Synthesized minimal ENNC ADD model: {args.output} ({len(data):,} bytes)")
        return 0

    if args.command == "build-mlp-nnc":
        data = build_mlp_ennc_flatbuffer(
            dim=args.dim, target_hw=args.target_hw, schema_version=args.schema_version
        )
        info = validate_and_inspect_model(data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(data)
        print_compatibility_report(info, "ENNC")
        print(
            f"\n  Synthesized ENNC MLP model (dim={args.dim}, TargetHw={args.target_hw}): "
            f"{args.output} ({len(data):,} bytes)"
        )
        return 0

    if args.command == "build-add-tflite":
        data = build_minimal_add_flatbuffer(b"TFL3", schema_version=3, target_hw=None)
        validate_and_inspect_model(data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(data)
        print(f"Synthesized minimal TFL3 ADD model: {args.output} ({len(data):,} bytes)")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
