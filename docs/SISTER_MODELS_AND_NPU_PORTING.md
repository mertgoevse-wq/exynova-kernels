# Samsung Sister-Model & Rose `.nnc` Portierung auf Exynos 1580 (Galaxy A56)

## 1. Kern-Erkenntnis: Galaxy A55 (`ex8845` / `Rose`) -> Galaxy A56 (`Santa` / `s5e8855`) Binär-Kompatibilität

Die Systemanalyse auf `SM-A566B` liefert den direkten Nachweis über `tools/nnc_inspect.py` und `tools/nnc_sister_zoo.py`:
Samsung verwendet auf dem Exynos 1580 (`A56`) direkt Produktions-`.nnc`-Modelle, die laut eingebetteter `COMPILED_COMMAND`-Metadaten explizit für das Schwester-SoC **Exynos 1480 (`ex8845` / Galaxy A55)** mit `--soc-type Rose` und **NCP v26** kompiliert wurden:

| Modell auf `SM-A566B` | Größe | Eingebettetes `SOC_TYPE` / Command | Eingabe-Tensor | Ausgabe-Tensor | Status auf A56 ENN `2.3.30` |
|---|---|---|---|---|---|
| `SRIB_HumanSegVideoLite_INT8_V101_ex8845_enn225k.nnc` | `1.3 MiB` | `Quadra` / `ex8845` (`Rose`) | `[1, 4, 288, 288]` F32 | `[1, 2, 288, 288]` F32 | Läuft nativ auf `/dev/vertex10` (`~6.5–7.5 ms`) |
| `SRIB_SID_INT16_V018_ex8845_enn200b.nnc` | `53 MiB` | `Quadra` / `--soc-type Rose` (`ex8845`) | `[1, 3, 832, 832]` F32 | `[1, 1, 832, 832]` F32 | Im Studio integriert (Saliency / Depth) |
| `SRIBMidas_aiUPSCALER_2X_V210_FP16_rose.nnc` | `40 MiB` | `Quadra` / `--soc-type Rose` | `[1, 12, 256, 256]` F32 (SpaceToDepth 2x2) | `[1, 3, 1024, 1024]` F32 | Im Studio integriert (2X/4X Super-Res) |
| `SRIBMidas_aiDEBLUR_V140_INT16_rose.nnc` | `45 MiB` | `Quadra` / `--soc-type Rose` | `[1, 12, 768, 768]` F32 | `[1, 12, 768, 768]` F32 | NPU Bild-Entschärfung |
| `SRIBMidas_aiCLARITY_2X_V200_INT16_rose.nnc` | `50 MiB` | `Quadra` / `--soc-type Rose` | `[1, 12, 256, 256]` F32 | `[1, 3, 1024, 1024]` F32 | NPU Schärfe & Upscaler (`A55` Build-Pfad!) |
| `aifrc.nnc` | `11 MiB` | `Santa` (Exynos 1580 nativ!) | `[1, 8, 272, 480]` F32 | `[1, 24, 272, 480]` F32 | Native A56 NPU Frame-Interpolation |
| `denoise_model_e3100_832X1088X4_v9.5.nnc` | `25 MiB` | `Quadra` / `--soc-type Rose --chip_version EVT1` | `[1, 4, 832, 1088]` F32 | `[1, 12, 832, 1088]` F32 | NPU RAW Denoise (`TRANSPOSE` + `ENN_NPU`) |

## 2. Praktische Schlussfolgerung für Apps & Hybrid-KI

1. **Keine kryptografische Signaturprüfung:** Eigene `.nnc`-Dateien und Schwester-`.nnc`-Dateien anderer `NCP v26` (`Rose`/`Quadra`/`Santa`) Exynos-Geräte (Galaxy A55 `s5e8845`, S22/S24 FE Exynos, A56 `s5e8855`) werden von ENN `2.3.30` direkt ohne Signatursperre geladen — entscheidend ist ausschließlich eine **binär-kompatible `NCP v26` CMDQ-ISA-Version** (`Rose` / `Quadra` / `Santa`) und ein korrekter `ENNC v211` Wrapper.
2. **NPU + GPU Hybrid (`ExyNova Vision-Chat Bridge`):**
   Da `Qwen2.5-1.5B` auf der `Xclipse 540` GPU läuft (offenes Vulkan `Q4_0` + `Q8_0` Head), übernimmt die Samsung NPU im Hybrid-Modus die **echte Bildanalyse** (`SRIB_HumanSeg` / `SRIB_SID` / `aiUPSCALER` auf `/dev/vertex10`, typisch >10× schneller als CPU-Spatial-Filter) und übergibt den erkannten Bildkontext direkt an `Qwen2.5` auf der GPU.
