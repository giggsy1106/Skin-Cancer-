Skin Cancer Multimodal Classification
Combining Dermatoscopic Images + Clinical Metadata for Better Diagnosis
📌 About This Project

Skin cancer is one of the most common cancers worldwide. Early detection can significantly improve patient outcomes.

This project builds a deep learning system that classifies skin lesions into 7 diagnostic categories using:

🖼 Dermatoscopic images

📋 Clinical metadata (age, sex, lesion location)

Instead of relying only on images, we combine both visual and patient information to build a more robust and clinically meaningful model.

This project was developed as part of the DS606 Data Science Capstone at UMBC.

📊 Dataset

We use the HAM10000 dataset (Human Against Machine with 10,000 images).

The dataset contains 10,000 dermatoscopic images categorized into 7 classes:

Code	Diagnosis
akiec	Actinic keratoses
bcc	Basal cell carcinoma
bkl	Benign keratosis
df	Dermatofibroma
nv	Melanocytic nevi
mel	Melanoma
vasc	Vascular lesions
Example Dermatoscopic Images
4

We also utilize available metadata:

Patient age

Sex

Anatomical location

🎯 Project Goals

The main goals of this project are:

Perform exploratory data analysis (EDA)

Handle severe class imbalance

Train a strong image-based baseline model

Build a multimodal model (image + metadata)

Evaluate performance using balanced metrics

Improve model calibration

Analyze fairness across demographic groups

Provide interpretability using Grad-CAM

🏗️ Project Structure
skin-cancer-multimodal/
├── configs/                     # Hydra/YAML configuration files
│   ├── model/                   # EfficientNet, Swin, ConvNeXt configs
│   ├── training/                # Learning rates, schedulers, loss functions
│   ├── data/                    # Augmentation, preprocessing configs
│   └── experiment/              # Full experiment presets
│
├── src/
│   ├── data/
│   │   ├── dataset.py           # HAM10000Dataset class + multi-modal loader
│   │   ├── transforms.py        # Dermatoscopy-specific augmentations
│   │   ├── sampler.py           # Class-balanced & group-aware samplers
│   │   └── preprocessing.py     # Color constancy, hair removal, resizing
│   │
│   ├── models/
│   │   ├── image_encoder.py     # EfficientNet-B4, Swin-T, ConvNeXt-V2
│   │   ├── metadata_encoder.py  # Learned embeddings for clinical metadata
│   │   ├── fusion.py            # Late fusion, FiLM, cross-attention
│   │   ├── classifier.py        # Final classification head
│   │   └── ensemble.py          # Model ensembling strategies
│   │
│   ├── training/
│   │   ├── trainer.py           # Main training loop with W&B logging
│   │   ├── losses.py            # Focal loss, cost-sensitive cross-entropy
│   │   └── scheduler.py         # Cosine annealing with warm restarts
│   │
│   ├── evaluation/
│   │   ├── metrics.py           # Balanced accuracy, macro F1, per-class AUC
│   │   ├── calibration.py       # Temperature scaling, reliability diagrams
│   │   ├── fairness.py          # Demographic subgroup analysis
│   │   └── statistical.py       # McNemar, DeLong tests, bootstrapping
│   │
│   └── interpretability/
│       ├── gradcam.py           # Grad-CAM / Grad-CAM++ visualizations
│       ├── attention.py         # Attention rollout for transformers
│       └── shap_analysis.py     # SHAP for metadata branch
│
├── notebooks/
│   ├── 01_EDA.ipynb             # Exploratory data analysis
│   ├── 02_Baseline.ipynb        # Image-only baseline experiments
│   ├── 03_MultiModal.ipynb      # Multi-modal fusion experiments
│   ├── 04_Ablation.ipynb        # Ablation study results
│   └── 05_Analysis.ipynb        # Final analysis, figures, model card
│
├── app/
│   └── gradio_demo.py           # Interactive inference demo
│
├── scripts/
│   ├── train.py                 # Main training entry point
│   ├── evaluate.py              # Evaluation pipeline
│   └── prepare_data.py          # Data download & preprocessing
│
├── model_card.md                # Google-format model card
├── requirements.txt             # Pinned dependencies
└── README.md                    # Project documentation


🧠 Model Overview
1️⃣ Image Encoder

We experiment with modern architectures such as:

EfficientNet-B4

Swin Transformer

ConvNeXt

These models extract high-level visual features from dermatoscopic images.

2️⃣ Metadata Encoder

Clinical metadata is encoded using:

Embedding layers

Fully connected neural networks

This allows the model to incorporate patient context into predictions.

3️⃣ Fusion Strategy

We explore different ways to combine image and metadata features:

Late fusion (concatenation)

Feature-wise modulation (FiLM)

Cross-attention mechanisms

This multimodal approach improves minority class detection, especially melanoma.

📈 Evaluation Metrics

Because the dataset is imbalanced, we focus on:

Balanced Accuracy

Macro F1-score

Per-class ROC-AUC

Confusion Matrix

We also evaluate:

Model calibration (temperature scaling)

Reliability diagrams

Demographic fairness (age/sex analysis)

🔍 Model Interpretability

In medical AI, interpretability is critical.

We implement Grad-CAM to visualize which regions of the lesion influence predictions.

Example visualization:

4

This helps ensure the model focuses on the lesion area rather than background artifacts.

📊 Expected Results
Model	Balanced Accuracy	Macro F1
Image-only Model	~0.80–0.83	~0.75–0.80
Multimodal Model	~0.85–0.88	~0.82–0.86

The multimodal setup improves melanoma detection and overall robustness.

⚠️ Limitations

Limited dataset size (10k images)

Metadata is incomplete

No external validation dataset

Not approved for clinical deployment

This project is strictly for research and educational purposes.

🚀 Future Improvements

External validation on ISIC dataset

Self-supervised pretraining

Better data augmentation

Uncertainty estimation

Web deployment as a decision-support tool

🛠️ Tech Stack

PyTorch

Hydra

Weights & Biases

NumPy / Pandas

Scikit-learn

SHAP

OpenCV
