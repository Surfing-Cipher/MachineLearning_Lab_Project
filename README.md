# Face Mask Detection using MobileNetV2

## 📌 Project Overview
This project implements a **Face Mask Detection System** using Deep Learning. It utilizes **MobileNetV2** as a base model for Transfer Learning to classify images into two categories:
*   `with_mask`
*   `without_mask`

The model is lightweight and efficient, suitable for real-time applications.

## 📂 Dataset
*   **Total Images**: ~5,381 images
    *   **Training Set**: 3,866 images
    *   **Testing Set**: 1,515 images
*   **Classes**:
    *   `with_mask` (Majority class)
    *   `without_mask` (Minority class)
*   **Preprocessing**:
    *   Resize to **224x224**
    *   Normalization (1./255)
    *   Data Augmentation (Rotation, Zoom, Horizontal Flip)

## 🏗 Model Architecture
*   **Base Model**: MobileNetV2 (Pre-trained on ImageNet)
    *   `include_top=False`
    *   `weights="imagenet"`
*   **Custom Head**:
    *   GlobalAveragePooling2D
    *   Dense (128 units, ReLU)
    *   Dropout (0.5)
    *   Dense (1 unit, Sigmoid) - Binary Classification

## ⚙️ Training Configuration
The model was trained in two phases to ensure stability and better performance.

### Phase 1: Initial Training (Transfer Learning)
*   **Base Model**: Frozen
*   **Optimizer**: Adam (`learning_rate=1e-4`)
*   **Loss Function**: Binary Crossentropy
*   **Epochs**: 5
*   **Performance**: Reached ~88.7% Validation Accuracy

### Phase 2: Fine-Tuning
*   **Unfreezing**: Unfroze layers after layer 100
*   **Optimizer**: Adam (`learning_rate=1e-5`) - Lower LR for stability
*   **Epochs**: 3
*   **Final Performance**: Reached **~90.1% Validation Accuracy**

## 📊 Evaluation Results
The model was evaluated on the test set using standard classification metrics.

*   **Overall Accuracy**: 66%
*   **Confusion Matrix**:
    *   Correctly predicted `with_mask`: **929**
    *   Correctly predicted `without_mask`: **65**
*   **Classification Report**:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **with_mask** | 0.79 | 0.78 | 0.78 | 1198 |
| **without_mask** | 0.19 | 0.21 | 0.20 | 317 |

> **Note**: The discrepancy between high validation accuracy during training (90%) and lower test accuracy (66%) suggests the test set distribution might be difficult or simpler metrics masked the class imbalance issue during training validation. The model is highly biased towards detecting masked faces.

## 🚀 Usage
1.  **Install Dependencies**:
    ```bash
    pip install tensorflow opencv-python matplotlib seaborn scikit-learn
    ```
2.  **Run the Notebook**:
    Open `ML_Project.ipynb` in your notebook environment and run the cells sequentially.

## 🔮 Future Improvements
*   Address class imbalance using class weights or oversampling.
*   Train for more epochs with Early Stopping.
*   Experiment with other architectures (ResNet, EfficientNet).
