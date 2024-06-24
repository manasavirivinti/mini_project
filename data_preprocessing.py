# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# def load_and_preprocess_data():
#     # Load dataset
#     data = pd.read_csv('network_security_data.csv')  # Replace with actual dataset path

#     # Handle missing values if any
#     data.fillna(method='ffill', inplace=True)

#     # Split features and target
#     X = data.drop(columns=['target'])
#     y = data['target'].apply(lambda x: 1 if x == 'anomaly' else 0)  # Convert target to binary labels

#     # Normalize features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     # Split data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#     dimensions = X_train.shape[1]
#     return X_train, X_test, y_train, y_test, dimensions
# import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

data = []
with open('network_security_data.csv', 'r') as file:
    for line in file:
        try:
            data.append(pd.Series(line.strip().split(',')))
        except Exception as e:
            print(f"Error processing line: {e}")

df = pd.DataFrame(data)


def load_and_preprocess_data():

    
    # Load dataset
    data = pd.read_csv('network_security_data.csv')  # Replace with actual dataset path

    # Separate features and target
    X = data.drop(columns=['target'])
    y = data['target']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Get the number of features (dimensions)
    dimensions = X_train.shape[1]

    return X_train, X_test, y_train, y_test, dimensions
