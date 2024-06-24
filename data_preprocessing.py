import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data():
    # Load dataset
    data = pd.read_csv('network_security_data.csv')  # Replace with actual dataset path

    # Handle missing values if any
    data.fillna(method='ffill', inplace=True)

    # Encode categorical features
    categorical_columns = ['Protocol', 'Service', 'Flag']
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # Split features and target
    X = data.drop(columns=['Label'])
    y = data['Label']

    # Encode the target variable
    y = LabelEncoder().fit_transform(y)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    dimensions = X_train.shape[1]
    return X_train, X_test, y_train, y_test, dimensions

# Example usage
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, dimensions = load_and_preprocess_data()
    print("Data loaded and preprocessed successfully.")
