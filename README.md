# Dermatology AI Diagnostic Assistant

This project implements a neural network-based diagnostic tool for differential diagnosis of erythemato-squamous diseases. The tool analyzes clinical symptoms to predict the most likely dermatological condition, aiming to reduce the need for unnecessary skin biopsies.

## About

Erythemato-squamous diseases share similar clinical features (erythema and scaling), making differential diagnosis challenging. This AI assistant helps distinguish between six conditions:

1. Psoriasis
2. Seboreic Dermatitis
3. Lichen Planus
4. Pityriasis Rosea
5. Cronic Dermatitis
6. Pityriasis Rubra Pilaris

## Dataset

The model is trained on the [Dermatology Dataset](https://www.kaggle.com/datasets/olcaybolat1/dermatology-dataset-classification) from Kaggle, which contains clinical and histopathological features of dermatology patients. This implementation uses only the 12 clinical features for diagnosis.

## Requirements

- Python 3.7+
- TensorFlow/Keras
- NumPy

## Installation & Usage

1. Download the required files:
   - `dermatology.py`
   - `requirements.txt`
   - `dermatology_model.weights.h5`

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the diagnostic tool:
   ```bash
   python dermatology.py
   ```

4. Follow the prompts to enter clinical symptoms (scale 0-3) and patient age.

## How It Works

The tool will:
- Ask for 12 clinical features based on a 0-3 scale
- Analyze the symptoms using a trained neural network
- Provide a diagnosis if confidence is ≥70%
- Recommend consulting a dermatologist and further biopsies (if needed) if confidence is <70%

## Disclaimer

This tool is for educational and research purposes only. It should not replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.
