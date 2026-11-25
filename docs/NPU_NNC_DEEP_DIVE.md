# Deep Dive: Was unterscheidet Samsung `.nnc` (`ENNC` + `NCP v26`) von eigenen `.tflite` / `.exyllm` Modellen?

Dieser Bericht dokumentiert direkt beobachtete Fakten aus:
- den drei gezogenen Samsung-`.nnc`-Produktionsmodellen auf dem Galaxy A56 (`/vendor/etc/...`),
- Live-`ENN_FRAMEWORK`-Logcat-Ausgaben während `EnnOpenModel` & `EnnExecuteModel`,
- sowie dem Samsung-Exynos-Kerneltreiber `third_party/a55-kernel/drivers/vision/npu/include/ncp_header_v26.h`.

---

## 1. Gibt es eine kryptografische Signaturprüfung in `.nnc`?

**Nein.** Unsere Live-Experimente mit `add_gpu_hw2.nnc` und `converted_add.nnc` auf `SM-A566B` beweisen in `ENN_FRAMEWORK`:
- Der Vendor-Dienst (`vendor.samsung_slsi.hardware.enn_aidl-service`) lädt selbst erzeugte `.nnc`-Dateien sofort, gibt `NNC(v2) identified` aus, liest `graph_version = 211`, erkennt `Operator[0] (hw: 2) ADD` und startet direkt `GpuUserDriver::OpenSubGraph(+)` inklusive OpenCL-Initialisierung (`Samsung Mobile GPU Platform`).
- Das bedeutet: **Es gibt keine Signatur- oder Zertifikatsprüfung auf `.nnc`-Dateien**, die vorgetäuscht oder umgangen werden müsste!

## 2. Warum scheiterten eigene `.tflite` / synthetische `.nnc` bislang?

Es gibt **zwei völlig unterschiedliche Gründe**, je nach `TargetHw`-Wert in `Operator` VTable-Slot `9` (`0x16`):

### A) `TargetHw = 4` (oder `ENN_NPU`, Op `1013`) → Echte NPU (`NpuUserDriver` / `/dev/vertex10`)
- Ein Blick in `SRIB_HumanSegVideoLite_INT8_V101_ex8845_enn225k.nnc` zeigt:
  - `Op[0]` (`ENN_PREPROC_1012`), `Op[1]` (`QUANTIZE`), `Op[3]` (`DEQUANTIZE`), `Op[4]` (`RESHAPE`) haben `TargetHw = 1` (CPU/ENN).
  - Nur `Op[2]` (`ENN_NPU`, Builtin-Code `1013`, `bopt_type = 127` = `ENN_NPUOptions`) läuft mit `TargetHw = 4` über `NpuUserDriver`.
  - Als zweiter Eingabetensor hängt daran `Tensor[6] "NPU_BINARY"` (`buffers[1]`, `1,38 MiB`) mit einem **fertigen `NCP v26`-Hardware-Programm** (`magic1 = 0x0C0FFEE0`, `ncp_ver = 26`, `magic2 = 0xC0DEC0DE`, `hdr_size = 848 B`, `body_size = 456.912 B`).
- Laut `ncp_header_v26.h` enthält dieser Body im `interruption_vector` die **kompilierten Maschinencode-Instruktionen (`cmdq isa`)** (`isa_offset`, `isa_size`, `weight_offset`, `const_offset`) sowie DMA-/LLC-Speichertabellen (`address_vector`, `memory_vector`, `MEMORY_TYPE_CUCODE`, `hw_cfu`) speziell für das proprietäre **Rose/Quadra NPU-ISA**.
- **Warum kein eigenes LLM (`Qwen2.5`, `SmolLM2`) allein auf der NPU läuft:** Samsungs Offline-Compiler (`nncontainer-gen` / `NPUC` / `AI LiteCore`), der einen Graphen in diese binären Rose/Quadra-CMDQ-Instruktionen (`NCP v26`) kompiliert und statische Tensorgrößen fest verdrahtet, ist nicht öffentlich dokumentiert und fehlt auf dem Gerät. Ohne diesen maschinenspezifischen CMDQ-Bytecode gibt es keinen offenen ISA-Assembler für `NCP v26` — und dynamische Sequenzlängen/KV-Caches eines Chat-LLMs werden vom statischen Rose-NPU-Compiler ohnehin auf die GPU verlagert.

### B) `TargetHw = 1` (ENN CPU) / `TargetHw = 2` (ENN OpenCL GPU) — **Durchbruch auf `SM-A566B`**
- Unsere frühere Testdatei `add_gpu_hw2.nnc` kam bis `GpuUserDriver::open_oplist start`, stürzte dort aber ab, weil:
  1. `Operator.builtin_options` (Slot `4`) leer war (`bopt_type = 0, has_bopt = False`), obwohl echte `.nnc`-Dateien für jeden Operator immer die passende `Options`-Tabelle (z. B. `AddOptions` `bopt_type = 1`) mitliefern,
  2. `Tensor` nur 2D `[1, 4]` statt 4D NCHW (`[1, 1, 1, 4]`) und ohne die `ENNC v211` Felder (`prev_operators`, `next_operators`, Slot `5 = 1`, Slot `11 = 100`) aufgebaut war.
- **Nach dem Fix in `tools/model_to_nnc.py` (live verifiziert auf `SM-A566B`):**
  - `valid_add_cpu_hw1.nnc` (`TargetHw = 1`, `656 B`, `ENNC v211`, mit `AddOptions` + 4D-Tensoren) **öffnet und läuft erfolgreich auf Samsung ENN `2.3.30` (`open_ms = 4.027 ms`, `exec_ms = 0.361 ms`, `ENN_RET_SUCCESS`, `EnnCloseModel Success`)**!
  - `valid_add_gpu_hw2.nnc` (`TargetHw = 2`) erreicht `A Model is created`, scheitert in `GpuUserDriver` aber weiterhin an OpenCL-Kern-Metadaten — und genau dafür haben wir **unsere eigene Vulkan-RDNA-Compute-Engine (`VulkanComputeBackend`)** geschrieben, die **direkt auf Xclipse 540** ohne `GpuUserDriver` läuft und echte Chatbots (`Qwen2.5-0.5B`, `SmolLM2-135M`, `TinyStories-15M`) sowie Spiele-KI (`Flappy AI`, `Neon Swarm 128x KI`) in Echtzeit antreibt!
