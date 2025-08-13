# 📌 **Project Overview**
This project focuses on building a **deep learning model** to classify **Indian food images** into **7 categories** using **PyTorch** and **EfficientNet-B0**.  
The model leverages **transfer learning** for high accuracy with limited training data, and includes **data augmentation** to improve generalization.

---

# 🚀 **Key Features**
- **Automatic dataset extraction & preprocessing** from a zipped archive.
- **Train/Test split** with balanced class distribution.
- **Image transformations** including resizing, normalization, horizontal flip, and rotation.
- **Transfer learning with EfficientNet-B0**, fine-tuned for 7 food classes.
- **Advanced training optimizations**:
  - AdamW optimizer
  - Cosine Annealing learning rate scheduler
  - Label smoothing in cross-entropy loss
- **Performance metrics:** Accuracy, Precision, Recall.

---

# 📂 **Dataset**
The dataset consists of **7 Indian food categories**, organized in `train/` and `test/` directories after preprocessing.  
Each category contains **hundreds of images** in `.jpg`, `.png`, and `.jpeg` formats.

---

# 🛠️ **Tech Stack**
- **Language:** Python 3
- **Libraries:** PyTorch, Torchvision, TorchMetrics, scikit-learn, Matplotlib
- **Model Architecture:** EfficientNet-B0 (pretrained on ImageNet)

---

# 📊 **Results**
After training for multiple epochs, the model achieved:
- **High classification accuracy** on unseen test data.
- **Strong generalization** due to augmentation and transfer learning.
- **Clear separation of classes** in visual predictions.

---
