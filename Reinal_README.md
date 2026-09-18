# 🩺 Retinal OCT Image Classification Using Deep Learning

A deep learning-based **multi-class retinal OCT image classification project** using Transfer Learning with multiple pretrained CNN architectures.

The goal of this project is to classify retinal Optical Coherence Tomography (OCT) images into **8 different retinal disease/condition classes** using ImageNet-pretrained convolutional neural networks.

---

## 📌 Project Overview

**Retinal OCT Image Classification** uses deep learning and transfer learning to classify retinal OCT images into eight different categories.

The project compares several state-of-the-art CNN architectures and evaluates their performance using multiple classification metrics and visualizations.

### Classification Classes

The model classifies images into the following 8 classes:

1. **DR** — Diabetic Retinopathy
2. **AMD** — Age-related Macular Degeneration
3. **CSR** — Central Serous Retinopathy
4. **DRUSEN**
5. **CNV** — Choroidal Neovascularization
6. **NORMAL**
7. **MH** — Macular Hole
8. **DME** — Diabetic Macular Edema

---

## 🧠 Models Used

The project compares six pretrained CNN architectures from TensorFlow/Keras Applications:

* **Xception**
* **ResNet50**
* **MobileNetV3Large**
* **EfficientNetB7**
* **DenseNet121**
* **InceptionV3**

All models use **ImageNet pretrained weights** and the original classification head is removed using:

```python
include_top=False
```

A custom classification head is then added for the 8-class retinal OCT classification task.

---

## 🔄 Transfer Learning Architecture

The general model pipeline is:

```text
Retinal OCT Image
        ↓
Resize to 224 × 224
        ↓
Pretrained CNN Backbone
        ↓
Global Average Pooling
        ↓
Batch Normalization
        ↓
Dropout
        ↓
Dense Layer (256 units)
        ↓
Dropout
        ↓
Softmax Output
        ↓
8 Retinal Classes
```

The pretrained backbone is initially frozen during Stage-1 transfer learning.

---

## ⚙️ Configuration

The notebook uses the following main configuration:

| Parameter                 |                    Value |
| ------------------------- | -----------------------: |
| Image Size                |                224 × 224 |
| Batch Size                |                       32 |
| Random Seed               |                       42 |
| Initial Epochs            |                        5 |
| Fine-Tuning Epochs        |                       10 |
| Output Classes            |                        8 |
| Optimizer                 |                   Adamax |
| Initial Learning Rate     |                     1e-4 |
| Fine-Tuning Learning Rate |                     1e-5 |
| Loss Function             | Categorical Crossentropy |

---

## 📂 Dataset Structure

The notebook expects the dataset to be organized into separate training, validation and testing directories.

```text
RetinalOCT_Dataset/
│
├── train/
│   ├── DR/
│   ├── AMD/
│   ├── CSR/
│   ├── DRUSEN/
│   ├── CNV/
│   ├── NORMAL/
│   ├── MH/
│   └── DME/
│
├── val/
│   ├── DR/
│   ├── AMD/
│   ├── CSR/
│   ├── DRUSEN/
│   ├── CNV/
│   ├── NORMAL/
│   ├── MH/
│   └── DME/
│
└── test/
    ├── DR/
    ├── AMD/
    ├── CSR/
    ├── DRUSEN/
    ├── CNV/
    ├── NORMAL/
    ├── MH/
    └── DME/
```

The notebook performs an image count check for each class across the training, validation and test datasets.

---

## 🛠️ Data Preprocessing

Images are resized to:

```text
224 × 224 × 3
```

Training images use augmentation including:

* Rotation
* Width shifting
* Height shifting
* Zoom
* Horizontal flipping

The training data generator uses:

```python
rotation_range=15
width_shift_range=0.10
height_shift_range=0.10
zoom_range=0.10
horizontal_flip=True
```

Validation and test generators are used without training augmentation.

---

## 🚀 Project Workflow

The notebook follows this workflow:

```text
1. Import Libraries
        ↓
2. Check TensorFlow / GPU
        ↓
3. Load Dataset
        ↓
4. Check Dataset Distribution
        ↓
5. Visualize Sample Images
        ↓
6. Create Data Generators
        ↓
7. Build Transfer Learning Models
        ↓
8. Train Multiple CNN Models
        ↓
9. Plot Training Accuracy/Loss
        ↓
10. Evaluate Models
        ↓
11. Classification Report
        ↓
12. Confusion Matrix
        ↓
13. ROC Curve
        ↓
14. Compare Model Metrics
        ↓
15. Identify Model for Further Tuning
        ↓
16. Fine-Tuning
        ↓
17. Final Evaluation
```

---

## 📊 Model Evaluation

Multiple evaluation metrics are used to analyze model performance.

### Accuracy

Measures the overall proportion of correctly classified images.

### Precision

Measures how many predicted positive samples are actually correct.

### Recall

Measures how many actual positive samples are correctly identified.

### F1-Score

Provides a combined measure of precision and recall.

The notebook calculates:

```text
Accuracy
Precision
Recall
F1-Score
```

using weighted averaging for the multi-class problem.

---

## 📈 Visualizations

Several visualizations are generated throughout the project.

### 1. Class Distribution

The number of images in each class is visualized across:

* Training
* Validation
* Testing

### 2. Sample Images

Sample OCT images are displayed together with their corresponding class labels.

### 3. Training Accuracy & Loss

Training and validation performance are visualized for the trained models.

### 4. Accuracy Comparison

The accuracy of all models is compared using a visualization.

### 5. Precision Comparison

Precision scores are compared across the different CNN architectures.

### 6. Recall Comparison

Recall performance is visualized for all models.

### 7. F1-Score Comparison

F1-score comparison provides a balanced view of classification performance.

### 8. Confusion Matrix

A confusion matrix is generated for each model to analyze class-wise predictions.

### 9. Classification Report

A detailed classification report is generated containing:

```text
Precision
Recall
F1-score
Support
```

for all eight classes.

### 10. ROC Curve

A multiclass ROC curve is generated using one-vs-rest class probabilities.

AUC values are calculated for individual classes.

### 11. Model vs Metrics Heatmap

A heatmap is used to compare:

```text
Model × Accuracy × Precision × Recall × F1
```

---

## 🔬 Fine-Tuning

After the initial transfer-learning stage, the project performs a second-stage fine-tuning process.

Initially:

```python
base_model.trainable = False
```

is used to freeze the pretrained backbone.

During fine-tuning:

```text
Pretrained CNN
      ↓
Unfreeze selected deeper layers
      ↓
Keep earlier layers frozen
      ↓
Use lower learning rate
      ↓
Continue training
```

The fine-tuning learning rate is reduced to:

```text
1e-5
```

This allows the pretrained network to adapt more carefully to the retinal OCT dataset.

---

## 🧪 Technologies Used

### Programming Language

* Python

### Deep Learning

* TensorFlow
* Keras

### Machine Learning

* Scikit-learn

### Data Processing

* NumPy
* Pandas

### Visualization

* Matplotlib
* Seaborn

### Environment

* Kaggle Notebook
* GPU acceleration

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/retinal-oct-image-classification.git
```

Move into the project directory:

```bash
cd retinal-oct-image-classification
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## 📋 Requirements

Example `requirements.txt`:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
jupyter
```

For GPU training, install a TensorFlow version compatible with your CUDA/GPU environment.

---

## ▶️ How to Run

### Option 1 — Kaggle

The notebook is designed to work well in a Kaggle GPU environment.

1. Open the notebook in Kaggle.
2. Add the Retinal OCT dataset.
3. Enable GPU.
4. Verify the dataset paths.
5. Run the notebook cells sequentially.

### Option 2 — Google Colab

Upload the notebook to Google Colab and configure the dataset path accordingly.

### Option 3 — Local Environment

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
Reinal-OCT-Image-Classification.ipynb
```

---

## 📁 Recommended Repository Structure

```text
retinal-oct-image-classification/
│
├── Reinal-OCT-Image-Classification.ipynb
│
├── README.md
│
├── requirements.txt
│
├── models/
│   ├── Xception/
│   ├── ResNet50/
│   ├── MobileNetV3Large/
│   ├── EfficientNetB7/
│   ├── DenseNet121/
│   ├── InceptionV3/
│   └── Xception_FineTuned/
│
├── visualizations/
│   ├── class_distribution/
│   ├── training_curves/
│   ├── confusion_matrix/
│   ├── roc_curves/
│   └── model_comparison/
│
└── dataset/
    └── RetinalOCT_Dataset/
```

> Dataset files should generally not be uploaded to GitHub if the dataset is large or has separate licensing/distribution restrictions.

---

## 📌 Important Notes

This project is intended for **educational and research purposes**.

The model predictions should **not be considered a medical diagnosis**. Real clinical diagnosis requires qualified ophthalmologists/medical professionals and appropriate clinical examination.

Model performance can also vary depending on:

* Dataset quality
* Class imbalance
* Image acquisition equipment
* Image preprocessing
* Training configuration
* Dataset distribution
* External/unseen data

Therefore, test-set performance should not automatically be interpreted as clinical performance.

---

## 🎯 Project Objectives

The main objectives are:

* Apply deep learning to retinal OCT image classification.
* Understand transfer learning using pretrained CNNs.
* Compare multiple CNN architectures.
* Evaluate multi-class classification performance.
* Analyze class-wise errors using confusion matrices.
* Study ROC performance.
* Visualize model performance.
* Perform second-stage fine-tuning.
* Build a reproducible medical-image classification workflow.

---

## 🔮 Future Improvements

Possible extensions include:

* More extensive hyperparameter tuning
* Class-weighted training
* Advanced data augmentation
* Cross-validation
* Ensemble learning
* Grad-CAM explainability
* External dataset validation
* Model calibration
* Precision-Recall curves
* Per-class ROC comparison
* TensorFlow Lite conversion
* FastAPI deployment
* Streamlit web application
* Docker deployment
* Cloud deployment

---

## 📊 Evaluation Summary

The project evaluates the six transfer-learning architectures using:

| Evaluation            | Included |
| --------------------- | -------- |
| Accuracy              | ✅        |
| Precision             | ✅        |
| Recall                | ✅        |
| F1-Score              | ✅        |
| Classification Report | ✅        |
| Confusion Matrix      | ✅        |
| ROC Curve             | ✅        |
| AUC                   | ✅        |
| Model Comparison      | ✅        |
| Metrics Heatmap       | ✅        |
| Fine-Tuning           | ✅        |

---

## 🏆 Model Comparison

The notebook compares:

```text
Xception
ResNet50
MobileNetV3Large
EfficientNetB7
DenseNet121
InceptionV3
```

The final model selection should be based on the evaluation results produced by the notebook rather than assuming that one architecture will always perform best.

---

## 👨‍💻 Author

**Md Emon Islam**

Deep Learning | Computer Vision | Machine Learning | AI Engineering

---

## ⭐ If You Find This Project Useful

If this project helps you learn about retinal OCT classification and transfer learning, consider starring the repository and using the notebook as a foundation for further research and experimentation.

