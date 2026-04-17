# Analformer: Transformer for both analysis and prediction

Analformer is a novel Transformer-based architecture designed for high-accuracy brain-computer interface (BCI) prediction while maintaining inherent neuroscientific interpretability. Unlike traditional "black box" deep learning models, Analformer bridges the gap between performance and explainability, allowing researchers to verify whether predictions are based on genuine neural activity.

## Key Innovation: Analytical Patch Embedding

The core of Analformer is its **Analytical Patch Embedding** module. It utilizes fixed, non-trainable **Morlet wavelet kernels** to extract explainable spatio-temporal-frequency features from raw EEG signals. 

This unique structure enables:
- **Direct Interpretation**: Direct, model-based generation of standard neurophysiological analyses (time-frequency, topography, and FTF analysis).
- **Functional Connectivity**: Inference of attention-based functional connectivity by analyzing the Transformer's attention mechanism applied to interpretable features.
- **Trustworthy AI**: Verification of model predictions against established neuroscientific findings.

---

## Technical Requirements (Prerequisites)

To run the research notebooks, ensure you have the following environment set up:

### System Requirements
- Python 3.8+
- PyTorch (CUDA supported version recommended, e.g., CU121)
- GPU with 8GB+ VRAM (for efficient training)

### Required Libraries
```bash
pip install torch torchvision torchaudio
pip install mne==1.9.0 mne-connectivity
pip install einops torchsummary scikit-learn seaborn mat73 h5py
```

---

## Repository Structure

This repository contains the following **refactored research notebooks** optimized for production and sharing:

- `clean_conformer_model_finetuning_down_250_K_MI_v18_F-value_att_ch_1_anal.ipynb`: 
    - Focused on **Motor Imagery (MI)** paradigm with 62-channel OpenBCI data.
- `clean_conformer_model_finetuning_down_250_K_ERP_v18_F-value_att_ch_1_anal.ipynb`: 
    - Specialized for **Event-Related Potential (ERP)** analysis.
- `clean_conformer_model_finetuning_down_250_K_SSVEP_v18_F-value_att_ch_1_anal.ipynb`: 
    - Optimized for **SSVEP** classification and analysis.
- `clean_conformer_finetuning_comp4_2a_v18_F-value_att_ch_1_anal.ipynb`: 
    - Implementation for the **BCI Competition IV 2a** dataset (22-channel).

---

## Data Preparation

The notebooks expect dataset paths to be configured in the `params` dictionary within the `main()` function.

1. **Dataset format**: `.mat` (HDF5 compatible) files.
2. **Channel Layout**: 
    - 62 channels for OpenBCI datasets.
    - 22 channels for BCI Competition 2a.
3. **Sampling Rate**: Resampled to 250Hz.

---

## How to Use

1. **Configure Parameters**: Open the desired notebook and locate the `params` dictionary in the `main()` function. Update the `"root"` path to your local data directory.
2. **Pre-training**: Execute the `Pretraining(params)` phase to train the model on multi-subject data.
3. **Fine-tuning**: Run the `Finetuning_and_Evaluation(...)` phase for subject-specific adaptation and performance evaluation.
4. **Neurophysiological Analysis**: Setting `"anal": 1` in `params` will trigger the automatic generation of:
    - **Time-Frequency Maps** (Morlet Wavelet)
    - **F-value Topographies**
    - **Attention-based Topographies**
    - **Functional Connectivity Graphs**

---

## Reference

If you use this code or work in your research, please refer to the following paper:

> **Analformer: Transformer for both analysis and prediction**  
> Hong Gi Yeom, Woo Sung Choi, and Kyung-min An.  
> *Scientific Reports* (2025).

---
*Note: All notebooks have been refactored for readability and shared research use. The core class names have been updated to `Analformer` and `analytical_patch_embedding` to reflect the latest research nomenclature.*