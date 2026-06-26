# Analformer: Eine neuartige Transformatorarchitektur für die EEG-Dekodierung und neurowissenschaftliche Analyse

<div align="center">
  <a href="README.md">English</a> |
  <a href="README.de.md"><strong>Deutsch</strong></a> |
  <a href="README.es.md">Español</a> |
  <a href="README.fr.md">français</a> |
  <a href="README.ja.md">日本語</a> |
  <a href="README.ko.md">한국어</a> |
  <a href="README.pt.md">Português</a> |
  <a href="README.ru.md">Русский</a> |
  <a href="README.zh.md">中文</a>
</div>

Dieses Repository bietet die offizielle Umsetzung des Artikels „Scientific Reports“:

**Eine neuartige Transformatorarchitektur für die EEG-Dekodierung und neurowissenschaftliche Analyse**

Hong Gi Yeom, Woo Sung Choi und Kyung-min An, *Scientific Reports* (2026).

DOI: [10.1038/s41598-026-56405-9](https://www.nature.com/articles/s41598-026-56405-9)

Analformer ist eine Transformer-basierte Architektur, die sowohl die hochleistungsfähige BCI-Dekodierung (Brain-Computer Interface) als auch interpretierbare neurowissenschaftliche Analysen unterstützt. Das Modell verwendet feste, nicht trainierbare Morlet-Wavelet-Kernel in einem Analytical Patch Embedding-Modul, sodass Zwischendarstellungen mit bekannten EEG-Analysetools wie Zeit-Frequenz-Karten, Topografien, F-Wert-Zeit-Frequenz-Analyse (FTF) und aufmerksamkeitsbasierter funktionaler Konnektivität analysiert werden können.

## Übersicht

Analformer wurde anhand öffentlicher EEG-Datensätze evaluiert, die vier Decodierungseinstellungen abdeckten:

- **BCI-Wettbewerb IV 2a**: Vier-Klassen-Motorbild (MI)
- **OpenBMI MI**: Zwei-Klassen-Motorbilder
- **OpenBMI ERP**: ereignisbezogene potenzielle Dekodierung mit zwei Klassen
- **OpenBMI SSVEP**: Steady-State-Decodierung visuell evozierter Potenziale mit vier Klassen

Die Kernidee besteht darin, die interpretierbare EEG-Struktur während der Einbettung zu bewahren. Feste Morlet-Wavelet-Filter extrahieren räumlich-zeitliche Frequenzmerkmale, der Transformer-Encoder lernt Beziehungen zwischen diesen Merkmalen und der Klassifizierungskopf sagt die Ziel-BCI-Klasse voraus. Analyseergebnisse werden aus den internen Darstellungen und Aufmerksamkeitsgewichtungen des Modells erstellt.

![Graphical abstract of Analformer](figures/Graphical_abstract.png)

## Repository-Struktur

- `Analformer_BCI_Comp_4_2a.ipynb`: BCI Competition IV 2a Motor Imagery-Implementierung.
- `Analformer_OpenBMI_MI.ipynb`: OpenBMI Motor Imagery-Implementierung.
- `Analformer_OpenBMI_ERP.ipynb`: OpenBMI ERP-Implementierung.
- `Analformer_OpenBMI_SSVEP.ipynb`: OpenBMI SSVEP-Implementierung.
- `figures/Graphical_abstract.png`: Grafische Zusammenfassung von Analformer.
- `figures/Fig4.png`, `figures/Fig5.png`, `figures/Fig7.png`: Beispiele für Analyseergebnisse aus dem Papier.
- `requirements.txt`: Von den Notebooks verwendete Python-Paketabhängigkeiten.

## Technische Anforderungen

Python 3.9+ wird empfohlen. Für das Training wird eine CUDA-fähige PyTorch-Installation empfohlen.

Installieren Sie zuerst PyTorch entsprechend Ihrer CUDA-Umgebung. Zum Beispiel:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Installieren Sie dann die verbleibenden Abhängigkeiten:

```bash
pip install -r requirements.txt
```

## Datenvorbereitung

Die Datensätze sind nicht in diesem Repository enthalten. Bitte laden Sie die öffentlichen Datensätze aus ihren Originalquellen herunter:

- BCI-Wettbewerb IV 2a: [http://www.bbci.de/competition/iv](http://www.bbci.de/competition/iv)
- OpenBMI: [https://gigadb.org/dataset/100542](https://gigadb.org/dataset/100542)

Jedes Notebook erwartet HDF5-kompatible `.mat`-Dateien mit `data`- und `label`-Arrays. Die Notebooks laden Betreffdateien nach dem folgenden Benennungsmuster:

- Trainingsdateien: `A1T.mat`, `A2T.mat`, ...
- Auswertungsdateien: `A1E.mat`, `A2E.mat`, ...

Legen Sie den Datensatzordner im `params`-Wörterbuch in `main()` fest:

```python
params = {
    "root": "Enter your dataset folder/",
    ...
}
```

Der Code geht von 250-Hz-EEG-Daten aus. Die OpenBMI-Notebooks nutzen 62 Kanäle, während das BCI Competition IV 2a-Notebook 22 Kanäle nutzt.

## Verwendung

1. Öffnen Sie das Notebook für den Datensatz/das Paradigma, den Sie ausführen möchten.
2. Aktualisieren Sie `"root"` im `params`-Wörterbuch in Ihren lokalen Datensatzordner.
3. Überprüfen Sie die wichtigsten Experimentparameter wie `n_ch`, `n_classes`, `time`, `baseline_sec`, `pretrain_epochs`, `finetuning_epochs`, `depth`, `num_heads` und `att_ch`.
4. Führen Sie die Notebook-Zellen der Reihe nach aus.
5. Verwenden Sie `"anal": 1`, um Analyseausgaben zu generieren, einschließlich Zeit-Frequenz-Karten, Topografien, F-Wert-Karten und aufmerksamkeitsbasierte Konnektivitätsvisualisierungen.

Für die OpenBMI-Notebooks wird `Pretraining(params)` vor der fachspezifischen Feinabstimmung ausgeführt. Im BCI Competition IV 2a-Notebook verwendet der aktuelle Workflow einen vorab trainierten Prüfpunktpfad, wenn das Vortraining übersprungen wird. Aktualisieren Sie `pretrained_path`, bevor Sie das Notebook ausführen.

## Beispiele für Analyseergebnisse

Die folgenden Abbildungen zeigen Beispiele repräsentativer Analyseergebnisse aus der Arbeit.

| Figure | Description |
| --- | --- |
| ![Fig4](figures/Fig4.png) | Topography analysis during the Motor Imagery (MI) task. |
| ![Fig5](figures/Fig5.png) | Whole-channel time-frequency analysis during the Motor Imagery (MI) task for Dataset II. |
| ![Fig7](figures/Fig7.png) | Attention-based Functional Connectivity analysis for the four Motor Imagery (MI) tasks in Dataset I. |

## Zitat

Wenn Sie diesen Code verwenden, geben Sie bitte Folgendes an:

```bibtex
@article{yeom2026analformer,
  title = {A novel transformer architecture for EEG decoding and neuroscientific analysis},
  author = {Yeom, Hong Gi and Choi, Woo Sung and An, Kyung-min},
  journal = {Scientific Reports},
  year = {2026},
  doi = {10.1038/s41598-026-56405-9}
}
```

## Danksagungen

Diese Arbeit wurde durch einen Forschungsfonds der Chosun-Universität unterstützt.

Teile dieser Implementierung wurden unter Bezugnahme auf das [EEG-Conformer](https://github.com/eeyhsong/EEG-Conformer)-Repository entwickelt, das unter der GPL-3.0-Lizenz vertrieben wird. Wir danken den Autoren, dass sie ihre Umsetzung öffentlich zugänglich gemacht haben.

## Lizenz

Der Quellcode wird unter der Lizenz in `LICENSE` vertrieben. Die Papierzahlen werden hier nur zur Erläuterung der offiziellen Umsetzung wiedergegeben; Einzelheiten zur Veröffentlichung und Informationen zur Figurenlizenzierung finden Sie auf der [Artikelseite](https://www.nature.com/articles/s41598-026-56405-9).
