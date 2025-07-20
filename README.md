# chess-selfplay-rl

Eine leichtgewichtige Selbstlern-Schach-KI, die sich (ähnlich AlphaZero/Leela Chess Zero) allein durch Selbstpartien kontinuierlich verbessert.  
Alle Komponenten sind in **Python 3 + PyTorch 2.2 (CUDA 12.8)** gehalten und laufen damit nativ auf einer GeForce **RTX 5070 (Blackwell-Architektur)**.

---

## 1. Motivation

* Reinforcement-Learning-basierte Engines (AlphaZero, Leela Chess Zero) haben gezeigt, dass allein *Selbstspiel* genügt, um ohne menschliche Partien superstarke Schach-AIs zu erzeugen.  
* Wir wollen **das Prinzip in minimaler, gut verständlicher Form** nachbauen, damit man jeden Baustein einzeln erweitern oder austauschen kann.

---

## 2. Voraussetzungen

| Komponente            | Empfehlung / Version | Hinweise |
|-----------------------|----------------------|----------|
| **GPU**               | NVIDIA GeForce RTX 5070 (GB205), Treiber ≥ 555.xx | Blackwell: Compute Capability 9.0, FP8-TensorCores |
| **CUDA**              | 12.8                 | Offiziell von PyTorch 2.2 unterstützt, viele Fixes für RTX 50-Serie |
| **Python**            | ≥ 3.11               | Verwendung von `venv` oder Conda |
| **PyTorch**           | 2.2.1 + cu128 Nightly | Enthält Flash-Attention-v2 & AOTInductor |
| **Weitere Pakete**    | `python-chess`, `tqdm`, `wandb`, `rich`, `numpy`, `omegaconf` |

> **Installation (einmalig)**  
> ```bash
> conda create -n chessai python=3.11
> conda activate chessai
> pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.2.1+cu128
> pip install python-chess tqdm wandb rich omegaconf
> ```

---

## 3. Projektstruktur

```

chess-selfplay-rl/
├─ README.md          ← diese Datei
├─ requirements.txt   ← Paketliste (s.o.)
├─ src/
│  ├─ config.yaml     ← Hyper-Parameter, Pfade
│  ├─ chess_zero/
│  │  ├─ model.py     ← ResNet-Policy/Value-Net (Skelett)
│  │  ├─ mcts.py      ← Monte-Carlo-Tree-Search
│  │  ├─ selfplay.py  ← Generiert Spiele gegen sich selbst
│  │  ├─ train.py     ← SGD/AdamW-Training aus Selfplay-Daten
│  │  ├─ evaluate.py  ← Spielstärke-Tests gegen Stockfish
│  │  └─ utils.py
│  └─ scripts/
│     ├─ setup_env.sh
│     ├─ run_selfplay.sh
│     └─ train.sh
└─ data/
├─ selfplay/…      ← .pt-Dateien (Bretter, Policy, Value)
└─ checkpoints/    ← gespeicherte Netze

```

---

## 4. Schnellstart

1. **Umgebung anlegen** (s.o.)  
2. **Skelett-Netz initialisieren**  
   ```bash
   python -m chess_zero.model --init checkpoints/000.pt
```

3. **Erste 100 Selbstpartien erzeugen**

   ```bash
   python -m chess_zero.selfplay \
          --games 100 \
          --model checkpoints/000.pt \
          --out data/selfplay
   ```
4. **Trainieren**

   ```bash
   python -m chess_zero.train \
          --data_dir data/selfplay \
          --checkpoint_out checkpoints/001.pt
   ```
5. **Spielen** (UCI-Modus gegen eine GUI deiner Wahl)

   ```bash
   python -m chess_zero.evaluate --engine checkpoints/001.pt --uci
   ```

---

## 5. Roadmap (High-Level)

| Phase | Ziel                       | Kernthemen                                                                                            | Erwartete Dauer |
| ----- | -------------------------- | ----------------------------------------------------------------------------------------------------- | --------------- |
| **0** | Umgebung & Skeleton        | Repo anlegen, Abhängigkeiten installieren, **Board-Encoding** implementieren                          | ½ Tag           |
| **1** | *Proof-of-Concept*         | Kleines 6×ResNet, vereinfachtes MCTS (max. 400 Simus/Zug), 5 000 Spiele                               | 2–3 Tage        |
| **2** | Stabiler Lern-Loop         | Daten-Pipeline (PyTorch Dataset), Checkpointing, ELO-Auswertung                                       | 1 Woche         |
| **3** | Skalierung                 | Größeres Netz (N=20), FP16/FP8 mixed-precision, Multi-GPU optional                                    | 2 Wochen        |
| **4** | Spielstärke > Stockfish 10 | Besseres Wert-/Politik-Netz, dynamische Temperatur, Resign-Erkennung                                  | 1–2 Monate      |
| **5** | Forschungsideen            | MuZero-artige Planung ohne exakten Zuggenerator, Curriculum-Learning, Transfer auf andere Brettspiele | offen           |

---

## 6. Erste TODO-Liste (für **Phase 0**)

| # | Aufgabe                                                                           | Datei         | Tipps                                         |
| - | --------------------------------------------------------------------------------- | ------------- | --------------------------------------------- |
| 1 | **Bitboard-Encoding**: 8×8×18 Boolean-Planes (6 Figuren × 2 Farben + 6 Aux-Infos) | `utils.py`    | Orientierung an AlphaZero-Paper ([GitHub][1]) |
| 2 | **ResidualBlock-Klasse** anlegen                                                  | `model.py`    | Beispielcode weiter unten                     |
| 3 | **MCTS-Gerüst**: Knoten-Klasse + Auswahlformel (PUCT)                             | `mcts.py`     | Starte mit `cpuct=2.0`, 100 Simulationen      |
| 4 | **Selfplay-Loop** (Worker ⇒ Spiel ⇒ `.pt`-Save)                                   | `selfplay.py` | Pro Spiel Begrenzung auf 160 Züge             |
| 5 | **config.yaml** mit Hyper-Parametern                                              | `config.yaml` | YAML erleichtert spätere Sweeps               |

```python
# Beispiel (src/chess_zero/model.py)
import torch, torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)
```

*(keine Sorge – diese fünf Tasks reichen, damit das Projekt “lebt”; alle Details wie Training-Loop, ELO-Berechnung usw. kommen Schritt für Schritt in den nächsten Iterationen.)*

---

## 7. Häufige Stolperfallen & Lösungen

* **CUDA-Mismatch** (RTX 50 ↔ PyTorch): Achte darauf, *genau* die `+cu128`-Builds zu nutzen; ältere `cu121`-Wheels liefern ⇒ *“unrecognized SM 90”*
* **Speicherbedarf**: Mixed-Precision (`torch.autocast(device_type="cuda", dtype=torch.float16)`) spart \~40 % VRAM.
* **Leistung**: Mit PyTorch 2.2 + AOTInductor + FlashAttention-v2 sind auf Blackwell-GPUs \~1.8× mehr Training-Samples/s drin als auf Ampere
* **Datenflut**: 1 000 Selbstpartien ≈ 50 MB; regelmäßiges Komprimieren verhindert SSD-Überlauf

---

## 8. Lizenz

MIT – frei für Forschung und Hobby-Projekte.

---

*Quellen:* AlphaZero-Open-Source Repo ([GitHub][1]) · Leela Chess Zero Quickstart ([Leela Chess Zero][6]) · RTX 50-Serie & Blackwell-Specs ([NVIDIA Newsroom][7]) · PyTorch 2.2 Release-Notes ([PyTorch Docs][4])
Weitere Links im Text vermerkt.

