# VDCNN for Mass Spectrometry Functional Group Classification

Our work presents a novel VDCNN architecture for accurate, high-throughput classification of functional groups from mass spectrometry data for nerve agents.
## Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA 11.2+ (for GPU acceleration)
### Quick Start
cd (your repository)
 **Install dependencies**
 pip install -r requirements.txt
 
 ## Usage

### Data Preparation
Place your mass spectrometry CSV files in the `data/` directory. Ensure they follow the format described in the `prepare_target_dataset` function within `src/utils.py`.

### Model Training
To reproduce the results from our paper, run:
python main.py

**Key Hyperparameters** (configurable in `src/main.py`):
- `kernel_size=5`
- `learning_rate=0.00005`
- `batch_size=64`
- `epochs=200`

### Evaluation
The training script automatically performs 10-fold cross-validation and reports:
- Weighted accuracy, precision, recall, and F1-score
- Per-class metrics with 95% confidence intervals

## Method Overview

### Architecture
- **Input**: Raw mass spectrometry spectra
- **Backbone**: 5-layer VDCNN with increasing filter sizes (64→512)
- **Classification head**: Global average pooling + fully connected layers
- **Output**: Multi-label predictions for functional groups

### Data Processing
- **Preprocessing**: Standardization and normalization
- **Class balancing**: Bootstrap resampling for imbalanced datasets

## Results & Reproducibility

Our VDCNN model achieves the following performance on the test set:

| Metric | Mean ± Std | 95% CI |
|--------|------------|---------|
| Accuracy | 94.2 ± 1.5 | [92.7, 95.7] |
| Precision | 93.8 ± 1.8 | [92.0, 95.6] |
| Recall | 94.2 ± 1.5 | [92.7, 95.7] |
| F1-score | 94.0 ± 1.6 | [92.4, 95.6] |

## Citation

If you use this code in your research, please cite our paper:




