# ExyNova einfach erklärt — Fachbegriffe ohne Vorwissen

Hier findest du alle wichtigen Wörter rund um KI auf dem Smartphone (Galaxy A56 / Exynos 1580) – einfach erklärt mit Beispielen aus dem Alltag.

---

## 1. CPU vs. GPU vs. NPU — Wer macht was im Handy?

Stell dir dein Smartphone wie eine große Großküche vor:

| Chip-Teil | Alltagsbeispiel | Was es besonders gut kann | Wie ExyNova ihn nutzt |
|---|---|---|---|
| **CPU** (Hauptprozessor) | **Chefkoch mit 8 Händen** – extrem schlau, kümmert sich um Ordnung, App-Steuerung & schwere Einzelaufgaben. | Viele verschiedene Regeln nacheinander abarbeiten. | Steuert die App, lädt Dateien, baut Prompts zusammen. |
| **GPU** (`Xclipse 540`) | **Große Brigade aus hunderten Hilfsköchen**, alle schneiden exakt gleichzeitig Karotten. | Tausende kleine Rechenschritte *gleichzeitig* (Spiele, Chatbot-Antworten berechnen). | Führt **Sprachmodelle (LLMs)** wie `SmolLM2` / `Qwen2.5`, Spiele (`Flappy AI`, `Neon Swarm`) per **Vulkan** aus. |
| **NPU** (`Samsung ENN`) | **Spezial-Maschine nur für fertige Tiefkühlpizzen** – blitzschnell & stromsparend, aber sie backt keine eigenen Rezepte. | Vorgebackene KI-Aufgaben (z. B. Kamera-Hintergrund erkennen) extrem effizient ausführen. | Führt echte Samsung-Systemmodelle (`.nnc`) auf `/dev/vertex10` aus – z. B. für Szenen-Erkennung & unseren Live-Beweis. |

---

## 2. Shader (Rechenprogramm für die GPU)

* **Was ist das?** Ein kleines Mini-Programm, das nicht auf der CPU läuft, sondern auf allen hunderten kleinen Kernen deiner GPU gleichzeitig.
* **Beispiel:** In einem LLM musst du Millionen Zahlen multiplizieren und addieren. Ein **Compute-Shader** (geschrieben in GLSL -> übersetzt in **SPIR-V**-Maschinencode) gibt jedem GPU-Kern ein kleines Stück dieser Liste, und alle rechnen im selben Moment.
* **Ohne Shader** müsste die CPU die Liste Stück für Stück allein abarbeiten.

---

## 3. Vulkan & SPIR-V

* **Vulkan** ist so etwas wie die **Steckdose / gemeinsame Sprache**, über die unsere App direkt mit der Samsung Xclipse 540 GPU spricht – ohne Umwege.
* **SPIR-V** ist die **komprimierte Anleitungskarte**, die Vulkan in Sekundenbruchteilen direkt an die GPU-Kerne schickt. ExyNova bringt 12 eigene, fertige SPIR-V-Karten mit (z. B. für Matrix-Multiplikation, RoPE, RMSNorm, Q4_0, Q8_0, Softmax, Conv2D).

---

## 4. Quantisierung (`FP32`, `FP16`, `Q8_0`, `Q4_0`)

KI-Modelle bestehen aus Millionen "Gewichten" (Nachkommastellen wie `0.38492017...`).

* **`FP32` (32-Bit):** Riesige Genauigkeit (wie jede Zutat aufs Millionstel Gramm abwiegen) – verbraucht sehr viel Speicher und Zeit.
* **`FP16` (16-Bit):** Halbe Größe – wie 2 Stellen nach dem Komma. Meist genau genug.
* **`Q8_0` (8-Bit):** Ganze kleine Zahlen (-127 bis +127) pro Block. ~4x kleiner als FP32!
* **`Q4_0` (4-Bit):** Nur 16 mögliche Abstufungen pro Gewicht (~8x kleiner als FP32!). Wie eine grobe Messbecher-Skala statt Laborwaage – für Chatbots immer noch erstaunlich klug, aber passt locker in den Handy-Speicher und läuft deutlich schneller.

---

## 5. `ELLM-Loader` & `.exyllm`-Dateien (`v1` / `v2` / `v3`)

* Modelle auf HuggingFace kommen in großen PC-Paketen (`.safetensors`).
* Der **`ELLM-Loader`** (`LlmEngine`) in ExyNova ist unser eigenes, schlankes Leseprogramm fürs Handy: Er liest unsere kompakte `.exyllm`-Datei, erkennt Architektur-Details (Schichten, `NeoX-RoPE`, `QKV-Bias` ab `v3`), lädt die komprimierten Gewichte einmalig in schnellen GPU-Speicher und merkt sich das bisherige Gespräch im **KV-Cache** (damit das Modell nicht bei jedem neuen Wort das ganze Gespräch von vorne durchrechnen muss!).

---

## 6. `.tok` (`ETOK`), BPE & `ChatML`

* Eine KI liest keine Wörter wie "Hallo", sondern zerlegt Text in **Tokens** (Wortbausteine wie `"Hal"` + `"lo"` -> Token-Nummern `#1204`, `#881`).
* **BPE (Byte-Pair Encoding):** Die Methode, wie seltene Wörter kleingeschnitten werden, sodass jede Sprache (auch Deutsch) darstellbar ist.
* **`ETOK` (`.exyllm.tok`):** Unsere handliche Wörterbuch-Datei passend zur `.exyllm`-Datei.
* **`ChatML`:** Ein einfaches Format mit Markierungen (`<|im_start|>user ... <|im_end|>`), damit ein Chat-Modell genau weiß: *"Ah, das hat der Mensch gefragt, jetzt bin ICH dran mit Antworten!"*

---

## 7. `.nnc`, `ENNC`, `NCP v26` & Warum Chatbots nicht einfach "NPU-only" laufen

* Samsung-Smartphones nutzen für Kamera-Effekte das Format **`.nnc`** (Magic-Bytes `ENNC`) über den **ENN-Dienst** (`libenn_user.so` -> `/dev/vertex10`).
* **Wichtige Erkenntnis aus unserer Analyse auf dem Galaxy A56:**
  * Samsung-`.nnc`-Dateien haben **keine kryptografische Signaturprüfung**, die eigene Dateien blockiert!
  * Warum laufen eigene `TFLite`- oder `Qwen`-Chatbots trotzdem nicht auf der NPU allein? Weil im Inneren einer echten NPU-`.nnc`-Datei (`TargetHw=4 / Op 1013 ENN_NPU`) ein **fertiges Maschinenprogramm im Format `NCP v26` (`CUCODE` / CMDQ-Befehle)** steckt. Das ist reiner Binärcode für Samsungs geschlossene NPU-Recheneinheit – vergleichbar mit einer fertigen Playstation-Disc. Den Offline-Compiler dafür (`nncontainer-gen`) gibt Samsung nicht öffentlich heraus.
  * Deswegen führt ExyNova **Chatbots ehrlich und komplett offen auf deiner Xclipse 540 GPU (Vulkan)** aus und beweist gleichzeitig die **echte NPU-Nutzung** über valide `.nnc`-Systemmodelle auf `/dev/vertex10` (ohne vorgetäuschte Fake-Ergebnisse).
