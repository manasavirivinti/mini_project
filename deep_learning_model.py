import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build_and_train_model(X_train, y_train, dimensions):
    # Define the model
    model = Sequential([
        Dense(64, input_dim=dimensions, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
    return model

def integrate_gjo_with_model(model, best_solution, X_train, y_train):
    # Use best_solution from GJO to initialize model weights
    initial_weights = model.layers[0].get_weights()
    print(f"Shape of initial weights[0]: {initial_weights[0].shape}")
    print(f"Size of best_solution: {best_solution.size}")

    initial_weights[0] = best_solution.reshape(initial_weights[0].shape)
    model.layers[0].set_weights(initial_weights)

    # Retrain the model with these weights
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
