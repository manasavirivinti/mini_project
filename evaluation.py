def evaluate_model(model, X_test, y_test):
    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Final Test Accuracy: {accuracy}')
