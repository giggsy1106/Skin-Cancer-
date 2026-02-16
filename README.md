2.1 Project Directory Structure
skin-cancer-multimodal/
├── configs/ # Hydra/YAML configuration files
│ ├── model/ # EfficientNet, Swin, ConvNeXt configs
│ ├── training/ # Learning rates, schedulers, loss functions
│ ├── data/ # Augmentation, preprocessing configs
│ └── experiment/ # Full experiment presets
├── src/
│ ├── data/
│ │ ├── dataset.py # HAM10000Dataset class + multi-modal loader
│ │ ├── transforms.py # Dermatoscopy-specific augmentations
│ │ ├── sampler.py # Class-balanced & group-aware samplers
│ │ └── preprocessing.py # Color constancy, hair removal, resizing
│ ├── models/
│ │ ├── image_encoder.py # EfficientNet-B4, Swin-T, ConvNeXt-V2
│ │ ├── metadata_encoder.py # Learned embeddings for clinical metadata
│ │ ├── fusion.py # Late fusion, FiLM, cross-attention
│ │ ├── classifier.py # Final classification head
│ │ └── ensemble.py # Model ensembling strategies
│ ├── training/
│ │ ├── trainer.py # Main training loop with W&B logging
│ │ ├── losses.py # Focal loss, cost-sensitive cross-entropy
│ │ └── scheduler.py # Cosine annealing with warm restarts
│ ├── evaluation/
│ │ ├── metrics.py # Balanced accuracy, macro F1, per-class AUC
│ │ ├── calibration.py # Temperature scaling, reliability diagrams
│ │ ├── fairness.py # Demographic subgroup analysis
│ │ └── statistical.py # McNemar, DeLong tests, bootstrapping
│ └── interpretability/
│ ├── gradcam.py # GradCAM / GradCAM++ visualizations
│ ├── attention.py # Attention rollout for transformers
│ └── shap_analysis.py # SHAP for metadata branch
├── notebooks/
│ ├── 01_EDA.ipynb # Exploratory data analysis
│ ├── 02_Baseline.ipynb # Image-only baseline experiments
│ ├── 03_MultiModal.ipynb # Multi-modal fusion experiments
│ ├── 04_Ablation.ipynb # Ablation study results
│ └── 05_Analysis.ipynb # Final analysis, figures, model card
├── app/
│ └── gradio_demo.py # Interactive inference demo
├── scripts/
│ ├── train.py # Main training entry point
│ ├── evaluate.py # Evaluation pipeline
│ └── prepare_data.py # Data download & preprocessing
├── model_card.md # Google-format model card
├── requirements.txt # Pinned dependencies
└── README.md # Project documentation
