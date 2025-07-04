# Chess RL mit PyTorch – zwei Phasen, von Menschen-Daten bis Self-Play

Dieses Projekt stellt ein zweiphasiges Trainings-Setup für ein Policy-Value-Netz für Schach bereit. Zunächst wird das Netz auf menschlichen Partien vortrainiert und anschließend durch selbstständiges Spielen skaliert.

## Voraussetzungen
- Python 3.9+
- CUDA 11.x
- 12 GB GPU-RAM

## Installation
```bash
pip install -r requirements.txt
```

## Nutzung
### Phase 1
```bash
bash scripts/run_phase1.sh
```
### Phase 2
```bash
bash scripts/run_phase2.sh
```

## Quickstart
1. Parse deine PGN-Daten mit `python data/pgn_parser.py <file.pgn>`.
2. Starte das Vortraining mit `bash scripts/run_phase1.sh` und gib deinen Datenpfad an.
3. Nutze den entstandenen Checkpoint für Phase 2 mittels `bash scripts/run_phase2.sh`.

## FAQ
- **Welche GPU wird empfohlen?** Eine RTX 5070 oder stärker mit mindestens 12 GB RAM.
- **Kann ich CPU-only trainieren?** Ja, allerdings deutlich langsamer. Setze `device: cpu` in der Konfiguration.

## Troubleshooting
- Stelle sicher, dass die Abhängigkeiten korrekt installiert sind (`pip install -r requirements.txt`).
- Bei Speicherproblemen Batch-Größe reduzieren oder Gradient Accumulation aktivieren.

## Projektstruktur
- `config/` – YAML-Konfigurationen für beide Trainingsphasen
- `data/` – Datenverarbeitung und vorbereitete Datensätze
- `src/models/` – Netzwerk-Definitionen
- `src/training/` – Trainingsroutinen, Logging und MCTS
- `scripts/` – Startskripte für beide Phasen

## Performance-Tipps
- Aktiviere Mixed Precision (AMP) für schnellere Berechnungen und geringeren Speicherbedarf
- Verwende `torch.backends.cudnn.benchmark = True` für optimierte Convolutionen
- Setze `pin_memory` und `persistent_workers` in den DataLoadern
- Teste unterschiedliche Batch-Größen entsprechend deines GPU-RAMs
- Nutze `torch.cuda.empty_cache()` beim Testen, um Speicher freizugeben
- Ab PyTorch 2.0 kann `torch.compile()` zusätzlichen Speed bringen

## Lizenz
Dieses Projekt steht unter der MIT License.
