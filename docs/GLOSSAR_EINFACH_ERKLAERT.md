# ExyNova Glossar — Alle Fachbegriffe einfach erklärt (mit Beispielen)

Dieses Dokument erklärt alle technischen Begriffe rund um **ExyNova**, **KI auf dem Smartphone**, **NPU & GPU**, **Flappy Bird AI** und **Quantisierung** ohne kompliziertes Kauderwelsch.

---

### 1. Shader & Compute-Shader
- **Was ist das?** Ein kleines Mini-Programm (oft nur 30–60 Zeilen), das direkt auf der **GPU (Grafikchip)** tausende Male **gleichzeitig** ausgeführt wird.
- **Alltags-Beispiel:** Stell dir vor, ein Lehrer (CPU) korrigiert 1.000 Klassenarbeiten einzeln nacheinander. Ein **Shader** ist, als würden 1.000 Lehrer gleichzeitig je genau **eine** Aufgabe in einer einzigen Sekunde korrigieren.
- **Bei ExyNova:** Unsere Dateien `.comp` (z. B. `matvec_q4_0.comp`) rechnen Blitzschnell Matrix-Zeilen für Sprachmodelle und Flappy-Bird-Entscheidungen auf der **Xclipse 540 GPU** aus.

### 2. SPIR-V
- **Was ist das?** Die fertige, kompakte Maschinensprache-Zwischenform, in die ein Shader übersetzt wird (wie eine MP3-Datei gegenüber Notenblättern).
- **Wozu gut?** Das Smartphone muss den Code nicht mühsam als Text neu übersetzen — Androids **Vulkan**-Treiber versteht SPIR-V sofort und startet ihn blitzschnell. `spirv-val` prüft vorab, ob jede einzelne Instruktion fehlerfrei ist.

### 3. GPU vs. NPU vs. CPU (Wer macht was im Galaxy A56?)
- **CPU (Prozessor):** Der Chefkoch. Extrem schlau bei komplizierten Ja/Nein-Entscheidungen (Dateien laden, Spiel-Logik, App-Menü), aber nur wenige Kerne (8 Kerne).
- **GPU (Xclipse 540 Grafikchip):** Eine riesige Mannschaft aus Rechenkernen. Ideal für Spiele-Grafik **und** große KI-Sprachmodelle (LLMs), weil LLMs vor allem aus Millionen einfacher Mal-und-Plus-Rechnungen bestehen.
- **NPU (Exynos Neural Processing Unit):** Ein reiner KI-Spezialchip. Noch sparsamer als die GPU, aber bei Samsung extrem spezialisiert auf fest kompilierte KI-Netze (z. B. Bild-Segmentierung wie `SRIB_HumanSegVideoLite...nnc`).

### 4. Quantisierung (`FP32`, `FP16`, `Q8_0`, `Q4_0`, `INT8`)
- **Was ist das?** KI-Modelle bestehen aus Millionen Gewichten (Zahlen). Standardmäßig braucht jede Zahl viel Speicher (**FP32** = 32 Bit, so genau wie 8 Nachkommastellen). **Quantisierung** rundet diese Zahlen geschickt auf kleinere Formate, sodass sie extrem wenig Speicher brauchen und die GPU sie viel schneller aus dem RAM laden kann!
- **Alltags-Beispiel:** Ein Foto in **RAW (40 MB)** vs. **JPEG (2 MB)**. Es sieht fast genauso gut aus, lädt aber 8× schneller.
- **Die Stufen bei ExyNova:**
  - **FP32 (32-Bit):** Maximale Präzision, größter Speicherbedarf (1×).
  - **FP16 (16-Bit):** Halbe Größe (2× kleiner), praktisch kein Qualitätsverlust.
  - **Q8_0 (8-Bit):** 4× kleiner als FP32 — jeweils 32 Zahlen teilen sich einen gemeinsamen Maßstab (*Scale*), super scharf.
  - **Q4_0 (4-Bit):** **8× kleiner** als FP32! Jede Zahl passt in nur ein halbes Byte (*Nibble*, 16 Stufen). Perfekt für schnelle lokale Sprachmodelle auf dem Smartphone.

### 5. ELLM-Loader (ExyNova LLM Loader)
- **Was ist das?** Unser eigens gebauter Modell-Lader für das kompakte Format `.exyllm`.
- **Was tut er?** Er öffnet die Modell-Datei, prüft den Header (`ELLM`), liest den Wortschatz (`.tok` BPE-Tokenizer) und schiebt alle Gewichte direkt in den schnellen GPU-Zwischenspeicher (**Weight Cache**), statt sie bei jedem Wort neu zu kopieren.

### 6. KV-Cache (Key-Value Cache)
- **Was ist das?** Das „Kurzzeitgedächtnis“ eines Chatbots während des Tippens.
- **Alltags-Beispiel:** Wenn du einen Satz Wort für Wort weiterliest, liest du nicht vor jedem neuen Wort das ganze Buch wieder von Seite 1 an. Genau das verhindert der KV-Cache: Er speichert bereits berechnete Aufmerksamkeits-Werte (Keys & Values) im Speicher, damit jedes weitere Wort sofort berechnet wird.

### 7. `.tflite` / `.nnc` / FlatBuffer / `TargetHw`
- **FlatBuffer:** Ein cleveres Speicherformat von Google, bei dem man Daten sofort lesen kann, ohne sie erst mühsam auszupacken (Zero-Copy).
- **`.tflite` (`TFL3`) → `.nnc` (`ENNC`):** Google speichert mobile Modelle als `.tflite`. Samsung nennt das nahezu identische Container-Format auf der NPU **ENNC** (`.nnc`). Unser Übersetzer schreibt die Kennung + Schema-Version (`211`) um und prüft das Feld **`TargetHw`** (Ziel-Chip: `1 = NPU`, `2 = GPU`), ob Samsung den fertigen Hardware-Maschinencode (**NCP v26**) bereits erwartet.

### 8. RMSNorm & RoPE
- **RMSNorm:** Hält die Rechenwerte zwischen den KI-Schichten in einer gesunden Größe (wie ein Lautstärke-Normalisierer bei Musik, damit nichts übersteuert oder zu leise wird).
- **RoPE (Rotary Position Embedding):** Dreht Zahlpaare leicht je nach Wort-Position. So weiß das KI-Modell den Unterschied zwischen *„Hund beißt Mann“* und *„Mann beißt Hund“*.

### 9. JNI & NDK
- **NDK (Native Development Kit):** Werkzeugkasten, mit dem man extrem schnellen C/C++-Code direkt für Android-Prozessoren kompiliert.
- **JNI (Java Native Interface):** Die „Übersetzer-Brücke“ zwischen unserer schicken Android-Oberfläche (Java) und der schnellen C++ / Vulkan / NPU Engine.

### 10. „Abliterated“ / Uncensored Modelle (für Sicherheitsforschung)
- **Was ist das?** Normale Chatbots haben eingebaute Verweigerungs-Regeln. Forscher nutzen die *Abliteration*-Technik (Entfernen genau der mathematischen Richtung im Netzwerk, die Pauschal-Ablehnungen auslöst), um **defensiv** testen zu können, ob externe Schutzfilter, Guardrails oder eigene System-Prompts gegen Prompt-Injections standhalten.
