import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import time

def load_data(file_path):
    data = pd.read_csv(file_path, delimiter=',', quotechar='"', on_bad_lines='skip')
    data = data.dropna()  # Consider handling missing values differently if necessary

    # Encode categorical columns
    categorical_columns = ['Protocol', 'Service', 'Flag']
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    X = data.drop(columns=['Label'])
    y = data['Label'].apply(lambda x: 1 if x == 'anomaly' else 0)
    return X.values, y.values

class GJO:
    def __init__(self, n_agents, n_features, n_iterations):
        self.n_agents = n_agents
        self.n_features = n_features
        self.n_iterations = n_iterations
        self.population = np.random.randint(0, 2, (n_agents, n_features))
        self.best_solution = None
        self.best_score = float('-inf')

    def fitness(self, solution, X, y):
        selected_features = np.where(solution == 1)[0]
        if len(selected_features) == 0:
            return 0
        X_selected = X[:, selected_features]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def update_population(self, X, y):
        for agent in range(self.n_agents):
            fitness_score = self.fitness(self.population[agent], X, y)
            if fitness_score > self.best_score:
                self.best_score = fitness_score
                self.best_solution = self.population[agent]

        for agent in range(self.n_agents):
            if np.random.rand() < 0.5:
                self.population[agent] = self.best_solution
            else:
                self.population[agent] = np.random.randint(0, 2, self.n_features)

    def optimize(self, X, y):
        for iteration in range(self.n_iterations):
            self.update_population(X, y)
        return self.best_solution

def build_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, activation='relu'), input_shape=input_shape))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    start_time = time.time()
    
    # Load data
    file_path = 'Train.csv'
    print("Loading data...")
    X, y = load_data(file_path)
    print("Data loaded.")
    load_time = time.time()
    
    # Feature selection using GJO
    print("Starting feature selection...")
    gjo = GJO(n_agents=10, n_features=X.shape[1], n_iterations=20)
    best_features = gjo.optimize(X, y)
    print("Selected Features:", np.where(best_features == 1)[0])
    gjo_time = time.time()
    
    X_selected = X[:, best_features == 1]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    # Reshape data for Bi-LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # Build and train model
    print("Building and training model...")
    model = build_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    train_time = time.time()
    
    # Predictions
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Classification report
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)
    
    # Total network attack percentage
    attack_percentage = (np.sum(y_pred) / len(y_pred)) * 100
    print(f"Total Network Attack Percentage: {attack_percentage:.2f}%")
    
    # Visualization
    plt.figure(figsize=(10, 5))
    plt.hist(y_pred, bins=2, color='blue', alpha=0.7)
    plt.title('Network Attack Detection')
    plt.xlabel('Attack (1) or No Attack (0)')
    plt.ylabel('Frequency')
    plt.xticks([0, 1], ['No Attack', 'Attack'])
    plt.show()
    
    # end_time = time.time()
    
    # # Print timing information
    # print(f"\nTiming Information:")
    # print(f"Data Loading and Preprocessing Time: {load_time - start_time:.2f} seconds")
    # print(f"GJO Feature Selection Time: {gjo_time - load_time:.2f} seconds")
    # print(f"Model Training Time: {train_time - gjo_time:.2f} seconds")
    # print(f"Total Execution Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
