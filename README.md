# AI-Powered Semiconductor Wafer Defect Detection System

**IESA DeepTech Hackathon 2026 – Phase 2 Submission**

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Why AI](#why-ai)
- [Phase 1: Model Development](#phase-1-model-development)
- [Phase 2: Inference & Compliance](#phase-2-inference--compliance)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Compliance Statement](#compliance-statement)
- [Technology Stack](#technology-stack)

---

## Problem Statement

Semiconductor wafer defect inspection is a critical quality control bottleneck in modern fabrication facilities. Manual inspection faces fundamental limitations:

- **Time Constraints**: Human inspectors require 3-5 minutes per wafer
- **Consistency Issues**: Fatigue-induced errors lead to detection rates varying between 60-75%
- **Scalability Limitations**: Manual processes cannot match modern fab throughput demands
- **Cost Overhead**: Specialized training and continuous monitoring require significant labor investment

This project addresses these challenges through automated, AI-powered defect classification using transfer learning and edge-optimized inference.

---

## Why AI

AI-powered defect detection offers significant advantages over traditional manual inspection:

**Technical Benefits**:
- Real-time inference capability (<50ms per image)
- Consistent multi-class defect classification
- Scalable deployment across production lines
- Edge-ready operation with no cloud dependency
- Adaptability to distribution shift through confidence thresholding

**Business Impact**:
- Reduces inspection time from minutes to milliseconds
- Eliminates human fatigue and subjective judgment
- Enables 24/7 automated quality control
- Lowers operational costs through automation
- Improves yield through early defect detection

---

## Phase 1: Model Development

### Dataset

**Training Data**: 2,066 images across 8 defect classes
- Train: 1,180 images (57%)
- Validation: 465 images (23%)
- Test: 421 images (20%)

**Defect Classes**:
1. **Bridge**: Unwanted electrical connections between circuit lines
2. **CMP Defect**: Chemical-mechanical polishing surface irregularities
3. **Cracks**: Structural fractures in wafer material
4. **LER (Line Edge Roughness)**: Irregular lithography pattern edges
5. **Opens**: Broken circuit connections
6. **Pattern Collapse**: Structural failure of photoresist patterns
7. **Undefected**: Clean wafer surfaces (baseline class)
8. **Via Defect**: Faulty vertical interconnect structures

### Model Architecture

```
Input: 224×224×3 RGB Image
    ↓
MobileNetV3Small (Pre-trained, Frozen)
    ↓
Global Average Pooling
    ↓
Dense Layer (Custom Classification Head)
    ↓
Softmax Output (8 classes)
```

**Architecture Details**:
- **Base Model**: MobileNetV3Small (ImageNet pre-trained)
- **Total Parameters**: 939K (5.7K trainable, 99.4% frozen)
- **Model Size**: 3.74 MB (ONNX format)
- **Optimization**: Transfer learning with frozen base layers

### Training Configuration

```python
Batch Size: 16
Epochs: 150 (early stopping)
Optimizer: Adam (lr=1e-3, ReduceLROnPlateau)
Loss: Categorical Crossentropy (label smoothing=0.1)
Regularization: Dropout (0.3-0.4), Frozen base model
Data Augmentation: Flips, rotations, brightness/contrast jitter
```

### Phase 1 Results

**Evaluation on Original Test Set (421 samples)**:

| Metric | Value |
|--------|-------|
| Test Accuracy | 78.04% |
| Weighted Precision | 80.00% |
| Weighted Recall | 78.00% |
| Weighted F1-Score | 78.00% |

**Model Export**:
- Format: ONNX (Open Neural Network Exchange)
- File: `wafer_defect_model.onnx`
- Validation: Max prediction difference < 0.000001 (Keras vs ONNX)

---

## Phase 2: Inference & Compliance

### Hackathon Constraints

Per Phase 2 rules, the following constraints were strictly enforced:

- **No Retraining**: Model weights remain unchanged from Phase 1
- **No Model Modification**: ONNX file used as-is without any alterations
- **Inference-Only**: Post-processing applied during inference, not training
- **Model Integrity**: `wafer_defect_model.onnx` remains byte-identical to Phase 1 submission

### Test Dataset Characteristics

**Hackathon Test Set**: 296 images across 8 classes

**Class Distribution**:

| Class | Count |
|-------|-------|
| bridge | 32 |
| cmp_defect | 30 |
| cracks | 31 |
| ler | 30 |
| opens | 30 |
| others | 80 |
| undefected | 33 |
| via_defect | 30 |

**Note**: The "others" class (80 images) includes samples not present during training, requiring the model to identify out-of-distribution data.

### Dynamic Class Loading

Classes are dynamically loaded from the test dataset directory structure:

```python
class_names = sorted(os.listdir(test_dir))
# Result: ['bridge', 'cmp_defect', 'cracks', 'ler', 'opens', 'others', 'undefected', 'via_defect']
```

This approach ensures:
- No hardcoded class lists
- Automatic adaptation to dataset structure
- Alphabetical ordering consistency

### Confidence-Based Post-Processing

**Strategy**: Low-confidence predictions are mapped to "others" class

```python
if max_probability < CONFIDENCE_THRESHOLD:
    prediction = "others"  # Low confidence
else:
    prediction = argmax(probabilities)  # High confidence
```

**Rationale**:
- Handles domain shift between training and test distributions
- Identifies out-of-distribution samples (true "others" class)
- Complies with hackathon requirement: "Mismatched classes expected to get classified into 'others'"
- Threshold set to 0.5 based on validation analysis

### Implementation

**Inference Script**: `src/hackathon_test_dataset_prediction.py`

**Key Components**:
- added particles class images to others as prescribed
- deleted pattern_collapse as no images are given by hackathon dataset
- ONNX Runtime for cross-platform inference
- Confidence threshold: 0.5
- Preprocessing: Resize to 224×224, normalize to [0,1]
- Dynamic class loading from dataset structure
- Comprehensive metrics computation
- Automated logging to results directory

### Phase 2 Results

**Final Evaluation Metrics**:

| Metric | Value |
|--------|-------|
| **Test Samples** | **296** |
| **Test Accuracy** | **29.73%** |
| **Precision (macro)** | **31.91%** |
| **Recall (macro)** | **31.66%** |
| **Correct Predictions** | **88 / 296** |
| **Confidence Threshold** | **0.5** |
| **Low-Confidence → "others"** | **151 / 296 (51.0%)** |

### Confusion Matrix

**8×8 Confusion Matrix** (Rows = True Labels, Columns = Predicted Labels):

```
True \ Pred         bridge    cmp_defect  cracks    ler       opens     others    undefected  via_defect  
------------------------------------------------------------------------------------------------------------
bridge              20        1           1         0         0         10        0           0         
cmp_defect          11        0           1         0         0         17        1           0         
cracks              1         8           5         0         0         15        0           2         
ler                 9         0           0         1         0         16        4           0         
opens               15        2           0         0         0         12        1           0         
others              15        8           3         1         1         50        2           0         
undefected          1         2           0         0         0         20        8           2         
via_defect          4         4           0         0         0         17        1           4         
```

### Per-Class Performance

| Class | Accuracy | Correct | Total | Mapped to "others" |
|-------|----------|---------|-------|--------------------|
| bridge | 62.5% | 20/32 | 32 | 10 |
| cmp_defect | 0.0% | 0/30 | 30 | 17 |
| cracks | 16.1% | 5/31 | 31 | 15 |
| ler | 3.3% | 1/30 | 30 | 16 |
| opens | 0.0% | 0/30 | 30 | 12 |
| others | 62.5% | 50/80 | 80 | 50 |
| undefected | 24.2% | 8/33 | 33 | 20 |
| via_defect | 13.3% | 4/30 | 30 | 17 |

### Performance Analysis

**Expected Performance Degradation**:

The accuracy drop from Phase 1 (78.04%) to Phase 2 (29.73%) is expected due to:

1. **Domain Shift**: Test data distribution differs significantly from training data
2. **Out-of-Distribution Class**: "others" class (80 images) not present during training
3. **No Adaptation**: Model cannot retrain or adapt to new distribution (per rules)
4. **Conservative Thresholding**: 0.5 threshold routes 51% of predictions to "others"

**Key Observations**:

- **Confidence Threshold Working**: 50/80 (62.5%) true "others" samples correctly identified
- **Domain Shift Impact**: Many legitimate classes fall below confidence threshold
- **Best Performing**: Bridge (62.5%) and "others" (62.5%) classes
- **Challenging Classes**: CMP Defect and Opens show 0% recall due to low model confidence

**Real-World Implications**:

This performance profile is typical for deployment scenarios where:
- Training and production data distributions differ
- Out-of-distribution samples must be detected
- Model cannot be retrained due to operational constraints
- Conservative predictions are preferred over false positives

### Generated Artifacts

Phase 2 inference generates three output files:

1. **`results/phase2_prediction_log.txt`**: Complete console output with timestamps
2. **`results/phase2_metrics.txt`**: Structured metrics summary
3. **`results/phase2_confusion_matrix.txt`**: Full 8×8 confusion matrix

---

## Repository Structure

```
IESA-DEEPTECH-AI-main/
│
├── dataset/
│   ├── train/                    # 1,180 training images (8 classes)
│   ├── val/                      # 465 validation images
│   └── test/                     # 296 hackathon test images (8 classes)
│
├── models/
│   ├── wafer_defect_model.keras  # Phase 1 trained model
│   └── wafer_defect_model.onnx   # ONNX export (UNCHANGED in Phase 2)
│
├── src/
│   ├── config.py                 # Configuration & paths
│   ├── data_loader.py            # Dataset loading & augmentation
│   ├── model.py                  # Model architecture
│   ├── train.py                  # Phase 1 training script
│   ├── evaluate.py               # Phase 1 evaluation
│   ├── export_to_onnx.py         # ONNX conversion
│   └── hackathon_test_dataset_prediction.py  # Phase 2 inference script
│
├── results/
│   ├── phase2_prediction_log.txt      # Full console output
│   ├── phase2_metrics.txt             # Structured metrics
│   └── phase2_confusion_matrix.txt    # Confusion matrix
│
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

---

## How to Run

### Prerequisites

```bash
pip install tensorflow>=2.13.0 onnxruntime scikit-learn opencv-python numpy
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

### Phase 2 Inference (Hackathon Submission)

**Run inference on test dataset**:

```bash
python src/hackathon_test_dataset_prediction.py
```

**Expected Output**:

```
Test samples: 296
Accuracy: 29.73%
Precision (macro): 31.91%
Recall (macro): 31.66%
Correct predictions: 88/296
Low-confidence → 'others': 151/296 (51.0%)
```

**Generated Files**:
- `results/phase2_prediction_log.txt`
- `results/phase2_metrics.txt`
- `results/phase2_confusion_matrix.txt`

### Phase 1 Training (Reference Only)

```bash
# Train model
python src/train.py

# Evaluate on original test set
python src/evaluate.py

# Export to ONNX
python src/export_to_onnx.py
```

---

## Compliance Statement

### Phase 2 Hackathon Rules Compliance

**Model Integrity**:
- ONNX model (`wafer_defect_model.onnx`) remains byte-identical to Phase 1 submission
- No retraining performed
- No weight modifications
- No architecture changes

**Inference Requirements**:
- Inference script named `hackathon_test_dataset_prediction.py`
- ONNX Runtime used for cross-platform inference
- Metrics computed: Accuracy, Precision (macro), Recall (macro), Confusion Matrix
- Console output logged to `phase2_prediction_log.txt`

**Post-Processing**:
- Confidence threshold (0.5) applied during inference only
- Low-confidence predictions mapped to "others" class
- No modifications to model predictions before threshold application

**Dynamic Class Handling**:
- Classes loaded dynamically using `sorted(os.listdir(test_dir))`
- No hardcoded class lists
- Automatic adaptation to dataset structure

### Model Verification

Verify ONNX model integrity:

```bash
certutil -hashfile models\wafer_defect_model.onnx SHA256
```

### Technical Specifications

| Specification | Value |
|---------------|-------|
| Model Format | ONNX |
| Model Size | 3.74 MB |
| Inference Engine | ONNX Runtime |
| Input Shape | (224, 224, 3) |
| Output Classes | 8 |
| Confidence Threshold | 0.5 |
| Preprocessing | Resize, Normalize [0,1] |
| Class Loading | Dynamic (sorted directory listing) |

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning Framework | TensorFlow 2.13+ / Keras |
| Model Architecture | MobileNetV3Small (Transfer Learning) |
| Model Export | ONNX (Open Neural Network Exchange) |
| Inference Engine | ONNX Runtime |
| Image Processing | OpenCV, NumPy |
| Evaluation | Scikit-learn |
| Language | Python 3.11 |

---

## Team & Acknowledgments

**Event**: IESA DeepTech Hackathon 2026  
**Domain**: Semiconductor Manufacturing / Industrial AI  
**Submission**: Phase 2 (Inference & Compliance)

**Special Thanks**: IESA for organizing this challenge in advancing AI applications for semiconductor manufacturing quality control.

---

## License

This project is developed for educational and hackathon purposes. For commercial use, please contact the team.

---

**Phase 2 Submission Complete** | Model: UNCHANGED | Inference: ONNX Runtime | Compliance: VERIFIED
