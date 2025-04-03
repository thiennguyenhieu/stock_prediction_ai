import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
from sklearn.model_selection import train_test_split
import os

# Function to load and preprocess the dataset
def load_and_preprocess_data(file_path, look_back=120, predict_days=30):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    
    df = pd.read_csv(file_path)
    
    # Check if necessary columns are in the dataframe
    required_columns = ['time', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'ROE (%)', 'ROA (%)', 'P/E', 'P/B', 'EPS (VND)', 'BVPS (VND)']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns. Ensure the dataset contains {', '.join(required_columns)}.")
    
    # Extract features and target
    features = df[required_columns].values
    target = df['close'].shift(-predict_days).dropna()  # Predict next 30 days closing prices
    
    # Normalize features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Create the dataset with look-back
    def create_dataset(data, target, look_back, predict_days):
        X, y = [], []
        for i in range(len(data) - look_back - predict_days):
            X.append(data[i:(i + look_back)])
            y.append(target[i + look_back: i + look_back + predict_days])  # 30-day future closing prices
        return np.array(X), np.array(y)

    X, y = create_dataset(features_scaled, target, look_back, predict_days)

    return X, y, scaler

# Function to create the CNN-LSTM model
def build_cnn_lstm_model(input_shape, predict_days):
    model = Sequential()

    # CNN layer
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))

    # LSTM layer
    model.add(LSTM(units=50, return_sequences=False))

    # Dense layer (output layer to predict 30 days)
    model.add(Dense(units=predict_days))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to train and evaluate the model
def train_and_evaluate_model(X_train, y_train, X_test, y_test, model):
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")

    return model

# Function to make predictions
def make_predictions(model, X_test, scaler, predict_days):
    predictions = model.predict(X_test)

    # Reverse the scaling of predictions (just the predictions, not the whole feature set)
    predictions = scaler.inverse_transform(np.hstack((np.zeros((predictions.shape[0], X_test.shape[2] - 1)), predictions)))

    # Extract only the predicted 'close' values
    return predictions[:, -predict_days:]  # Only the last `predict_days` columns are predictions

# Execution function
def execute(file_path, look_back=120, predict_days=30):
    try:
        # Load and preprocess the data
        X, y, scaler = load_and_preprocess_data(file_path, look_back, predict_days)

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Build the model
        model = build_cnn_lstm_model((X_train.shape[1], X_train.shape[2]), predict_days)

        # Train and evaluate the model
        trained_model = train_and_evaluate_model(X_train, y_train, X_test, y_test, model)

        # Make predictions
        predictions = make_predictions(trained_model, X_test, scaler, predict_days)

        # Display first few predictions for the next 30 days
        print("Predictions (first 5 samples for the next 30 days):")
        for i, pred in enumerate(predictions[:5]):
            print(f"Sample {i + 1}: {pred}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Provide the path to your dataset here
    file_path = "trained_stock_data.csv"
    execute(file_path)
