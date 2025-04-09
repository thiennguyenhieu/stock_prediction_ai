import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout
import matplotlib.pyplot as plt

INPUT_SEQ_LEN = 30
EPOCHS = 50
BATCH_SIZE = 32

# --- Create Sequences ---
def create_sequences(data, input_seq_len, feature_cols, target_cols):
    X, y = [], []
    for i in range(len(data) - input_seq_len):
        x_seq = data.iloc[i:i+input_seq_len][feature_cols].values
        y_vals = data.iloc[i+input_seq_len][target_cols].values
        X.append(x_seq)
        y.append(y_vals)
    return np.array(X), np.array(y)

# --- Build Model ---
def build_cnn_lstm_model(input_shape, output_dim):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, 3, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(output_dim)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Sample line plots (first N samples)
def plot_predictions(y_true, y_pred, label, n=100):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true[:n], label=f"True {label}", linewidth=2)
    plt.plot(y_pred[:n], label=f"Predicted {label}", linestyle="--")
    plt.title(f"{label} – True vs Predicted")
    plt.xlabel("Sample Index")
    plt.ylabel(label)
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Main ---
def main():
    # Load your data
    df = pd.read_csv("data/stock_data.csv")
    
    # Scaling
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    df[INPUT_FEATURES] = scaler_X.fit_transform(df[INPUT_FEATURES])
    df[TARGET_COLS] = scaler_y.fit_transform(df[TARGET_COLS])

    # Create sequences
    X, y = create_sequences(df, INPUT_SEQ_LEN, INPUT_FEATURES, TARGET_COLS)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train model
    model = build_cnn_lstm_model(input_shape=(INPUT_SEQ_LEN, len(INPUT_FEATURES)), output_dim=len(TARGET_COLS))
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Predict
    predictions = model.predict(X_val)
    predictions_inv = scaler_y.inverse_transform(predictions)
    y_val_inv = scaler_y.inverse_transform(y_val)

    # Plot for each target column
    for i, col in enumerate(TARGET_COLS):
        print(f"\n--- MAE for {col} ---")
        mae = mean_absolute_error(y_val_inv[:, i], predictions_inv[:, i])
        print(f"{col}: {mae:.4f}")
        plot_predictions(y_val_inv[:, i], predictions_inv[:, i], label=col, n=100)

    # Save model
    os.makedirs("model", exist_ok=True)
    model.save("model/cnn_lstm_stock_model.h5")
    print("\nModel saved to model/cnn_lstm_stock_model.h5")

if __name__ == "__main__":
    main()
