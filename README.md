Code will be shown after the publication.

Analformer: Transformer for both analysis and prediction
This repository contains the official PyTorch implementation of Analformer, a Transformer-based model designed for electroencephalography (EEG) signal classification. The model leverages an analytical patch embedding strategy using fixed Morlet wavelets to create interpretable and physiologically relevant input for a standard Transformer encoder.

✨ Overview
Analformer is designed to address the unique challenges of EEG data by:

Analytical Feature Extraction: Instead of a fully learned embedding, it uses a non-trainable convolutional layer with Morlet wavelet kernels to extract robust time-frequency features. This reduces the number of trainable parameters and grounds the model in established signal processing techniques.

Interpretability: The model is built for analysis. It allows for easy visualization of intermediate features, attention maps, and their neurophysiological implications (e.g., connectivity, topography).

Two-Phase Training: The framework includes a pre-training phase on data from all subjects, followed by a subject-specific fine-tuning phase to achieve high performance in motor imagery (MI) and other EEG-based classification tasks.

📂 Project Structure
The project is organized into several modular files for clarity and maintainability:

analformer/
├── main.py             # 🚀 Main script: Configure parameters and run the experiment
├── model.py            # 🏛️ Model: Defines the Analformer PyTorch model architecture
├── engine.py           # ⚙️ Engine: Contains the training, evaluation, and analysis logic
├── utils.py            # 🛠️ Utilities: Includes visualization and other helper functions
└── requirements.txt    # 📋 Dependency list
🏛️ Model Architecture
The model consists of three main components, all defined in model.py:

AnalyticalPatchEmbedding: Processes raw EEG signals to create input patch sequences for the Transformer. It uses a temporal convolution with fixed Morlet wavelets to extract time-frequency features.

Transformer Encoder: A standard stack of Transformer blocks that learns relationships between patches via multi-head self-attention.

Classification Head: A simple Multi-Layer Perceptron (MLP) head that performs classification based on the final representation from the Transformer.

⚙️ Installation
To get started, clone the repository and set up the required environment.

Clone the repository:

Bash

git clone https://github.com/your-username/analformer.git
cd analformer
Create and activate a virtual environment (recommended):

Bash

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the dependencies from requirements.txt:

Bash

pip install -r requirements.txt
Note: Your system may require a specific PyTorch version depending on your CUDA installation. If you encounter issues, please install PyTorch separately first by following the official instructions.

📊 Dataset
The code is configured to work with the BCI Competition IV 2a dataset.

Format: The data for each subject should be preprocessed and saved as .mat files.

Structure: The script expects the data files to be located in the directory specified in main.py:

/path/to/your/dataset/
├── A01T.mat
├── A01E.mat
├── ...
└── A09T.mat
└── A09E.mat
.mat File Content: Each .mat file (e.g., A01T.mat) must contain two specific variables:

data: A 3D array containing the EEG signals. The dimensions must be (channels, timepoints, trials).

label: A 2D array for the corresponding labels. The dimensions must be (trials, classes), and the labels must be one-hot encoded.

🚀 Usage
The entire experiment is controlled and launched from main.py.

Configure Parameters: Open main.py and modify the params dictionary. The most critical parameter to set is the path to your dataset:

Python

# Inside main.py
params = {
    # --- Data and Path Configuration ---
    "root": "/path/to/your/dataset/", # IMPORTANT: Update this path
    # ... other parameters for training and model architecture
}
Run the script:
Execute main.py from your terminal. The modular structure is handled by Python's import system, so you only need to run this single file.

Bash

python main.py
The script will automatically perform:

Phase 1: Pre-training the model (logic in engine.py).

Phase 2: Fine-tuning and evaluating the model for each subject (logic in engine.py).

📈 Outputs
The script will create a results/ directory inside your specified root path.

Log Files: log_subject{n}.txt for each subject's epoch-by-epoch results and a summary log file.

Saved Models: pretrained_analformer.pth and model_sub{n}_best.pth for each subject.

Visualizations: If anal: 1 is set, various analysis plots (attention maps, t-SNE) will be saved. All visualization functions are located in utils.py.

📄 Citation
If you use this code or the Analformer model in your research, please cite our paper:

@article{Yeom_2025_Analformer,
  title={Analformer: Transformer for both analysis and prediction},
  author={Hong Gi Yeom, Woo Sung Choi, and Kyung-min An},
  journal={in progress},
  year={2025}
}

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
