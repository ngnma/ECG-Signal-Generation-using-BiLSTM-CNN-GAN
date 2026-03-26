# ECG Signal Generation using BiLSTM-CNN GAN

Implementation of a Generative Adversarial Network (GAN) for synthetic ECG signal generation, developed as part of a collaborative university project.

---

## 🚀 Overview

Electrocardiogram (ECG) signals are widely used in medical diagnosis, but access to large datasets is limited due to privacy concerns.

This project develops a GAN-based system that learns the distribution of real ECG signals and generates synthetic data with similar characteristics, without exposing sensitive patient information.

---

## 👥 Project Context

This work was completed as a **team-based academic project**, where the system was designed, implemented, and evaluated collaboratively.

---

## 🧠 Methodology

The system follows a structured pipeline:

1. Data preprocessing  
2. Model design  
3. Adversarial training  
4. Evaluation  

Each stage was developed and validated independently before integration.

---

## 📊 Dataset & Preprocessing

The model is trained on the PhysioNet/CinC 2017 dataset:

- ~8,000 ECG recordings across multiple classes  
- Converted into ~26,000 fixed-length signal segments  

Preprocessing includes:
- Signal filtering and noise reduction  
- Normalisation for consistent scale  
- Segmentation into fixed-length sequences  
- Train/test split designed to prevent data leakage  

---

## 🏗️ Model Architecture

### Generator (BiLSTM)

- 2 Bidirectional LSTM layers  
- Fully connected output layer with sigmoid activation  

**Input:** `(batch, sequence_length, noise_dim)`  
**Output:** `(batch, sequence_length, 1)`  

The generator models temporal dependencies to produce realistic ECG waveforms.

---

### Discriminator (CNN)

- 1D Convolutional Neural Network  
- Multiple convolutional layers followed by a classification head  

The discriminator learns local ECG patterns and outputs the probability of a signal being real or generated.

---

## ⚔️ Training

The model is trained using an adversarial framework:

- The generator produces synthetic ECG signals  
- The discriminator evaluates their realism  
- Both models are updated iteratively  

Different configurations were explored to balance the learning dynamics between the two networks.

---

## ⚙️ Experiments

Multiple configurations were tested, varying:

- Generator capacity (e.g., hidden units)  
- Discriminator architecture  
- Learning rate balance  

The best-performing configuration used a higher-capacity generator and balanced training between both models.

---

## 📈 Evaluation

Evaluation was performed on unseen test data using standard signal similarity metrics.

### Results

| Metric | Value |
|--------|------|
| RMSE   | 0.298 |
| PRD    | 63.89% |
| Frechet Distance | 0.358 |

All metrics met the predefined acceptance thresholds.

---

## 📊 Results & Insights

- The model successfully captures the overall structure of ECG signals  
- Basic heartbeat patterns (e.g., R-peaks) are learned  
- Fine-grained waveform details remain challenging  

### Key Insight
Model performance depends strongly on the balance between generator and discriminator, as well as training stability.

---

## 🛠️ Tech Stack

- Python  
- PyTorch  
- NumPy, SciPy  
- Pandas  
- Matplotlib  
- Scikit-learn  

---

## 📄 Reference

This project implements and adapts the methodology proposed in:

Zhu, F., Ye, F., Fu, Y., Liu, Q., & Shen, B. (2019).  
*Electrocardiogram generation with a bidirectional LSTM-CNN generative adversarial network*.  
Scientific Reports, 9, 6734.  
https://doi.org/10.1038/s41598-019-42516-z

This implementation was developed independently for academic purposes using a different dataset and experimental setup.
