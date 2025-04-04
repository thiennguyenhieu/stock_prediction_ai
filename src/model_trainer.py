import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from stock_config import PRICE_FEATURES, VALUATION_FEATURES, COL_TIME, COL_SYMBOL

def preprocess_stock_data(csv_path, time_steps=30):
    ALL_FEATURES = PRICE_FEATURES + VALUATION_FEATURES

    df = pd.read_csv(csv_path)
    df[COL_TIME] = pd.to_datetime(df[COL_TIME])
    df = df.sort_values(by=COL_TIME)
    df = df[[COL_TIME, COL_SYMBOL] + ALL_FEATURES]
    df = df.fillna(method='ffill').dropna()

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df[ALL_FEATURES])
    scaled_df = pd.DataFrame(scaled_values, columns=ALL_FEATURES)

    def create_sequences(data, time_steps):
        X, y_price, y_valuation = [], [], []
        for i in range(time_steps, len(data)):
            X.append(data[i-time_steps:i])
            y_price.append(data[i][0:len(PRICE_FEATURES)])
            y_valuation.append(data[i][len(PRICE_FEATURES):])
        return np.array(X), np.array(y_price), np.array(y_valuation)

    X, y_price, y_valuation = create_sequences(scaled_df.values, time_steps)
    return X, y_price, y_valuation, scaler, PRICE_FEATURES, VALUATION_FEATURES

def build_and_train_model(csv_path, time_steps=30, epochs=10, batch_size=32):
    X, y_price, y_valuation, scaler, PRICE_FEATURES, VALUATION_FEATURES = preprocess_stock_data(csv_path, time_steps)

    X_train, X_val, y_price_train, y_price_val, y_val_train, y_val_val = train_test_split(
        X, y_price, y_valuation, test_size=0.2, random_state=42
    )

    input_layer = Input(shape=(time_steps, len(PRICE_FEATURES) + len(VALUATION_FEATURES)))
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    shared = Dense(64, activation='relu')(x)
    price_output = Dense(len(PRICE_FEATURES), name='price_output')(shared)
    valuation_output = Dense(len(VALUATION_FEATURES), name='valuation_output')(shared)

    model = Model(inputs=input_layer, outputs=[price_output, valuation_output])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            "price_output": "mse",
            "valuation_output": "mse"
        },
        metrics={
            "price_output": "mae",
            "valuation_output": "mae"
        }
    )

    history = model.fit(
        X_train,
        {"price_output": y_price_train, "valuation_output": y_val_train},
        validation_data=(X_val, {"price_output": y_price_val, "valuation_output": y_val_val}),
        epochs=epochs,
        batch_size=batch_size
    )

    return model, history, scaler

if __name__ == "__main__":
    csv_path = "data\stock_data.csv"
    model, history, scaler = build_and_train_model(csv_path, time_steps=30, epochs=10, batch_size=32)
