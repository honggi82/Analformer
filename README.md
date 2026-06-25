# Analformer: Transformer for EEG Decoding and Neuroscientific Analysis

This repository provides the official implementation of the Scientific Reports article:

**A novel transformer architecture for EEG decoding and neuroscientific analysis**

Hong Gi Yeom, Woo Sung Choi, and Kyung-min An, *Scientific Reports* (2026).

DOI: [10.1038/s41598-026-56405-9](https://www.nature.com/articles/s41598-026-56405-9)

Analformer is a Transformer-based architecture designed to support both high-performance brain-computer interface (BCI) decoding and interpretable neuroscientific analysis. The model uses fixed, non-trainable Morlet wavelet kernels in an Analytical Patch Embedding module so that intermediate representations can be analyzed with familiar EEG analysis tools such as time-frequency maps, topographies, F-value time-frequency (FTF) analysis, and attention-based functional connectivity.

## Overview

Analformer was evaluated on public EEG datasets covering four decoding settings:

- **BCI Competition IV 2a**: four-class Motor Imagery (MI)
- **OpenBMI MI**: two-class Motor Imagery
- **OpenBMI ERP**: two-class event-related potential decoding
- **OpenBMI SSVEP**: four-class steady-state visually evoked potential decoding

The core idea is to preserve interpretable EEG structure during embedding. Fixed Morlet wavelet filters extract spatio-temporal-frequency features, the Transformer encoder learns relationships between these features, and the classification head predicts the target BCI class. Analysis outputs are produced from the model's internal representations and attention weights.

![Graphical abstract of Analformer](figures/Graphical_abstract.png)

## Repository Structure

- `Analformer_BCI_Comp_4_2a.ipynb`: BCI Competition IV 2a Motor Imagery implementation.
- `Analformer_OpenBMI_MI.ipynb`: OpenBMI Motor Imagery implementation.
- `Analformer_OpenBMI_ERP.ipynb`: OpenBMI ERP implementation.
- `Analformer_OpenBMI_SSVEP.ipynb`: OpenBMI SSVEP implementation.
- `figures/Graphical_abstract.png`: Graphical abstract of Analformer.
- `figures/Fig4.png`, `figures/Fig5.png`, `figures/Fig7.png`: Analysis result examples from the paper.
- `requirements.txt`: Python package dependencies used by the notebooks.

## Technical Requirements

Python 3.9+ is recommended. A CUDA-enabled PyTorch installation is recommended for training.

Install PyTorch first according to your CUDA environment. For example:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

The datasets are not included in this repository. Please download the public datasets from their original sources:

- BCI Competition IV 2a: [http://www.bbci.de/competition/iv](http://www.bbci.de/competition/iv)
- OpenBMI: [https://gigadb.org/dataset/100542](https://gigadb.org/dataset/100542)

Each notebook expects HDF5-compatible `.mat` files with `data` and `label` arrays. The notebooks load subject files using the following naming pattern:

- Training files: `A1T.mat`, `A2T.mat`, ...
- Evaluation files: `A1E.mat`, `A2E.mat`, ...

Set the dataset folder in the `params` dictionary inside `main()`:

```python
params = {
    "root": "Enter your dataset folder/",
    ...
}
```

The code assumes 250 Hz EEG data. The OpenBMI notebooks use 62 channels, while the BCI Competition IV 2a notebook uses 22 channels.

## How to Use

1. Open the notebook for the dataset/paradigm you want to run.
2. Update `"root"` in the `params` dictionary to your local dataset folder.
3. Check the key experiment parameters such as `n_ch`, `n_classes`, `time`, `baseline_sec`, `pretrain_epochs`, `finetuning_epochs`, `depth`, `num_heads`, and `att_ch`.
4. Run the notebook cells in order.
5. Use `"anal": 1` to generate analysis outputs, including time-frequency maps, topographies, F-value maps, and attention-based connectivity visualizations.

For the OpenBMI notebooks, `Pretraining(params)` is executed before subject-specific fine-tuning. In the BCI Competition IV 2a notebook, the current workflow uses a pretrained checkpoint path when skipping pre-training; update `pretrained_path` before running that notebook.

## Analysis Result Examples

The following figures show representative analysis result examples from the paper.

| Figure | Description |
| --- | --- |
| ![Fig4](figures/Fig4.png) | Attention-based functional connectivity for BCI Competition IV 2a MI classes. |
| ![Fig5](figures/Fig5.png) | Accuracy comparison across Transformer encoder depths and attention head counts. |
| ![Fig7](figures/Fig7.png) | Wavelet convolution and AnalNet analysis pipeline. |

## Citation

If you use this code, please cite:

```bibtex
@article{yeom2026analformer,
  title = {A novel transformer architecture for EEG decoding and neuroscientific analysis},
  author = {Yeom, Hong Gi and Choi, Woo Sung and An, Kyung-min},
  journal = {Scientific Reports},
  year = {2026},
  doi = {10.1038/s41598-026-56405-9}
}
```

## Acknowledgements

This work was supported by research fund from Chosun University. During code development, we referred to the excellent [EEG-Conformer](https://github.com/eeyhsong/EEG-conformer) repository, and we thank its authors for making their implementation publicly available.

## License

The source code is distributed under the license in `LICENSE`. The paper figures are reproduced here only to explain the official implementation; please refer to the [article page](https://www.nature.com/articles/s41598-026-56405-9) for publication details and figure licensing information.
