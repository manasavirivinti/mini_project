from data_preprocessing import load_and_preprocess_data
from gjo_algorithm import run_gjo
from deep_learning_model import build_and_train_model, integrate_gjo_with_model
from evaluation import evaluate_model

# Step 1: Data Preprocessing
X_train, X_test, y_train, y_test, dimensions = load_and_preprocess_data()

# Step 2: Run Golden Jackal Optimization (GJO)
best_solution = run_gjo(X_train, y_train, dimensions)

# Step 3: Build and Train the Deep Learning Model
model = build_and_train_model(X_train, y_train, dimensions)

# Step 4: Integrate GJO with the Deep Learning Model and Retrain
integrate_gjo_with_model(model, best_solution, X_train, y_train)

# Step 5: Evaluate the Model
evaluate_model(model, X_test, y_test)