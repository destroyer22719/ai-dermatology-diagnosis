import warnings

# Suppress warnings before any other imports
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+")
warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer")

import tensorflow as tf
from keras import layers, models, Input
import numpy as np

num_classes = 6

model = models.Sequential([
  Input(shape=(12,)),
  layers.Dense(32, activation='relu'),
  layers.Dropout(0.2),
  layers.Dense(64, activation='relu'),
  layers.Dropout(0.2),
  layers.Dense(32, activation='relu'),
  layers.Dropout(0.2),
  layers.Dense(num_classes, activation='softmax')
])

model.compile(
  loss='categorical_crossentropy',
  metrics=['accuracy']
)  

model.load_weights('dermatology_model.weights.h5')

CLASS_NAMES = [
    "psoriasis",
    "seboreic dermatitis",
    "lichen planus",
    "pityriasis rosea",
    "cronic dermatitis",
    "pityriasis rubra pilaris"
]

QUESTIONS = [
    ("Erythema (0-3): ", int, 0, 3),
    ("Scaling (0-3): ", int, 0, 3),
    ("Definite borders (0-3): ", int, 0, 3),
    ("Itching (0-3): ", int, 0, 3),
    ("Koebner phenomenon (0-3): ", int, 0, 3),
    ("Polygonal papules (0-3): ", int, 0, 3),
    ("Follicular papules (0-3): ", int, 0, 3),
    ("Oral mucosal involvement (0-3): ", int, 0, 3),
    ("Knee and elbow involvement (0-3): ", int, 0, 3),
    ("Scalp involvement (0-3): ", int, 0, 3),
    ("Family history (0 for no, 1 for yes): ", int, 0, 1),
    ("Age: ", int, 0, 120)
]

def get_user_input():
  """Gets and validates user input for all features."""
  symptoms = []
  print("Please enter the clinical features based on the scale provided:")
  for question, type_cast, min_val, max_val in QUESTIONS:
    while True:
      try:
        value = type_cast(input(question))
        if min_val <= value <= max_val:
          symptoms.append(float(value))
          break
        else:
          print(f"Invalid input. Please enter a number between {min_val} and {max_val}.")
      except ValueError:
        print("Invalid input. Please enter a valid number.")
  return np.array(symptoms).reshape(1, -1)

def main():
  """Main function to run the CLI application."""
  print("--- Dermatology AI Diagnostic Assistant ---")
  print("This tool is designed to assist in the differential diagnosis of erythemato-squamous diseases,")
  print("which share similar clinical features. By analyzing symptoms, it provides a likely diagnosis,")
  print("aiming to reduce the need for unnecessary skin biopsies.")
  print("\nThe model predicts one of the following conditions:")
  for i, name in enumerate(CLASS_NAMES, 1):
    print(f"{i}. {name.title()}")
  print("-" * 41)

  user_symptoms = get_user_input()
  
  # Predict
  prediction_probs = model.predict(user_symptoms, verbose=0)
  predicted_class_index = np.argmax(prediction_probs)
  confidence = prediction_probs[0][predicted_class_index] * 100
  
  print("\n--- Prediction Result ---")
  if confidence >= 70:
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    print(f"The predicted condition is: {predicted_class_name.title()}")
    print(f"Confidence: {confidence:.2f}%")
  else:
    print("The diagnosis is uncertain based on the provided symptoms.")
    print("Consulting a dermatologist for a professional evaluation and potentially a biopsy is recommended.")
  
  print("\nNote: This prediction is based on symptom analysis from a neural network trained on a dataset, and should not replace professional medical advice.")

if __name__ == "__main__":
  main()