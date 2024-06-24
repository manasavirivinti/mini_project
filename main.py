import numpy as np
from data_preprocessing import load_and_preprocess_data
from deep_learning_model import build_and_train_model, integrate_gjo_with_model
from gjo_algorithm import run_gjo

# Load and preprocess data
X_train, X_test, y_train, y_test, dimensions = load_and_preprocess_data()

# Build and train initial model
model = build_and_train_model(X_train, y_train, dimensions)

# Get initial weight shape from the model
initial_weight_shape = model.layers[0].get_weights()[0].shape
print("Initial weight shape:", initial_weight_shape)  # Verify the shape

# Run GJO to find the best solution
best_solution = run_gjo(X_train, y_train, initial_weight_shape)

# Integrate GJO solution with the model and retrain
integrate_gjo_with_model(model, best_solution, X_train, y_train)
