# Robust Persian Speech Emotion Recognition with Ensemble Learning 🎙️🧠

This project implements a **robust Speech Emotion Recognition (SER) system for Persian language** using the **ShEMO dataset**.  
It combines **Convolutional Neural Networks (CNNs)** trained on spectrograms with **LSTMs** trained on MFCC features, and integrates them using **ensemble learning** for improved robustness.

---

## 📌 Features
- Uses the **ShEMO Persian emotional speech dataset**.
- Preprocessing: Mel spectrograms, MFCCs, normalization, padding/truncation.
- Data augmentation: **Noise injection** + **SpecAugment** for better generalization.
- Two complementary models:
  - **CNN** (captures spectral patterns from Mel spectrograms)
  - **LSTM** (captures temporal dynamics from MFCCs)
- **Soft voting ensemble** for final classification.
- Handles **6 emotions**:
  - Anger, Sadness, Fear, Happiness, Surprise, Neutral

---

## 📊 Results
- **CNN alone**: moderate accuracy, strong on spectral patterns.
- **LSTM alone**: strong on sequential dynamics, weaker on noise.
- **Ensemble (CNN + LSTM)**:
  - **Test Accuracy:** ~69%
  - **F1-score:** ~66.8%

---

Download ShEMO dataset

You’ll need the ShEMO dataset from Kaggle:
ShEMO Persian Emotional Speech Dataset


Ensemble Learning

This project uses soft voting ensemble:

CNN predicts emotions from Mel spectrograms

LSTM predicts emotions from MFCCs

Final prediction = average of both probability distributions

This improves robustness and generalization in noisy, real-world speech.

Future Work

Deploy as a real-time emotion recognition API.

Extend to multi-lingual emotion recognition.

Improve accuracy with transformer-based models (e.g., wav2vec2, HuBERT).

Preprocessing & Augmentation

Mel Spectrogram Conversion: Each audio file is converted into a Mel spectrogram, which represents sound frequencies over time.

Normalization: Spectrograms are standardized (subtract mean, divide by std).

Fixed Length (126 frames ≈ 4 sec): Padding or truncation ensures uniform input size.

Noise Augmentation: Gaussian noise is added during training to simulate real-world audio conditions.

Class Weighting: Since emotions are imbalanced, class weights are computed to avoid bias.

3. Model 1 — CNN with SpecAugment

Input: Mel spectrograms

Architecture:

3 convolutional layers (32 → 64 → 128 filters)

BatchNorm + ReLU

MaxPooling layers

SpecAugment: masks random frequency & time sections of spectrograms → forces robustness

Global Average Pooling (GAP)

Dropout (0.25)

Fully connected layers (128 → 6 outputs)

Training details:

Optimizer: AdamW

Scheduler: CosineAnnealingLR

Loss: CrossEntropy with class weights

Early stopping: patience 15

Output: Probability distribution over 6 emotions.

4. Model 2 — LSTM with MFCC

Input: MFCC (Mel-Frequency Cepstral Coefficients), another speech feature.

Architecture:

Bidirectional LSTM

Hidden states aggregated

Fully connected layer → Softmax over 6 classes

Strength: Captures temporal dependencies (intonation, rhythm) better than CNNs.

5. Ensemble Learning

Both CNN (spectrogram-based) and LSTM (MFCC-based) are trained separately.

Their predictions are combined using soft voting (averaging predicted probabilities).

This reduces variance and improves generalization.

Final results:

Test Accuracy: ~69%

F1-score: ~66.8%

6. Why This Project is Robust

SpecAugment → prevents overfitting and increases noise robustness.

Noise Injection → simulates real-world audio conditions.

Balanced Class Weights → handles class imbalance.

Ensemble → CNN (good at spatial patterns) + LSTM (good at temporal patterns) complement each other.

🎯 Key Takeaways

This project demonstrates how to build a robust emotion recognition system in Persian.

Uses spectrogram + MFCC features for richer audio representation.

Employs CNN + LSTM ensemble for better accuracy.

Achieves ~69% accuracy, which is strong given the challenges of SER (human-level is also imperfect).
