# ASL Alphabet Classification

This repository implements four image classification pipelines for the **American Sign Language (ASL)** alphabet
recognition task. Each pipeline takes raw hand images and predicts one of the ASL letters (plus special symbols) using
different feature extraction and classification strategies.

---

## 🔍 Repository Structure

```
├── data/                       # Downloaded datasets
│   ├── asl_alphabet_dev/       # Original ASL alphabet images
│   ├── asl_alphabet_train/     # Original ASL alphabet images
│   └── synthetic_test/         # Out-of-distribution test images
├── models/                     # Saved trained models and pipelines
├── training/                   # Jupyter / Colab notebooks for traning the pipelines
│   ├── train_classical.ipynb
│   ├── train_torch.ipynb
│   ├── train_hybrid.ipynb
│   └── train_combined.ipynb
├── exploratory_data_analysis.ipynb
├── final_evaluation.ipynb
├── src/                        # Python source code
│   ├── BasePipeline.py     # Abstract base class
│   ├── ClassicalPipeline.py
│   ├── CombinedPipeline.py
│   ├── HybridPipeline.py
│   ├── TorchPipeline.py
│   └── utils.py                # Utility functions (metrics, plotting), datadownloading
├── README.md                   # This file
├── classical3.ipynb            # For finding the hyperparameters of the ClassicalPipeline
├── combi.ipynb                 # For finding the hyperparameters of the CombinedPipeline
├── google_landmark.ipynb       # For finding the hyperparameters of the HybridPipeline
├── resnet.ipynb                # For finding the hyperparameters of the TorchPipeline
├── resnet_eval.ipynb           # For evaluating the TorchPipeline
└── requirements.txt            # Python dependencies
```

---

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/00I0/math_ds_asl.git
   cd math_ds_asl
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download data**:

    * Download the appropriate datasets from kaggle
    * Split them into test dev and train sets
    * Place the appropriate directories under `data/`

---

## 🔧 Pipelines Summary

1. **ClassicalPipeline**: Grayscale → Resize → Flatten → PCA → StandardScaler → RandomForest
2. **TorchPipeline**: Transfer‑learned ResNet‑18 (fine‑tuned) with data augmentation and LR finder
3. **HybridPipeline**: MediaPipe Hands → 21 keypoint landmark coordinates → RandomForest
4. **CombinedPipeline**: Concatenate PCA features + landmark features → RandomForest

All pipelines implement a unified interface:

```python
pipe = PipelineClass().load(model_path)
pipe.train(train_dir)
metrics = pipe.evaluate(test_dir)
```

---

## Dependencies

* Python 3.11
* numpy, pandas, matplotlib, seaborn
* scikit-learn
* torch, torchvision
* mediapipe
* pillow
* tqdm

See full version in `requirements.txt`.

