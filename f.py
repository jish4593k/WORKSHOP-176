import tkinter as tk
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the California Housing dataset
california_housing = fetch_california_housing()
X = california_housing.data
y = california_housing.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features using scikit-learn
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Linear Regression model using scikit-learn
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Train a neural network regression model using TensorFlow/Keras
model = Sequential([
    Dense(8, input_dim=X_train_scaled.shape[1], activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)

# GUI Setup
def predict_linear_regression():
    input_values = [float(entry.get()) for entry in entry_widgets]
    scaled_input = scaler.transform([input_values])
    linear_prediction = linear_model.predict(scaled_input)
    linear_result.set(f"Linear Regression Prediction: ${linear_prediction[0]:,.2f}")

def predict_neural_network():
    input_values = [float(entry.get()) for entry in entry_widgets]
    scaled_input = scaler.transform([input_values])
    nn_prediction = model.predict(scaled_input)
    nn_result.set(f"Neural Network Prediction: ${nn_prediction[0][0]:,.2f}")

# Create the main window
root = tk.Tk()
root.title("AI Regression GUI")

# Add entry widgets for input values
entry_widgets = []
for i, feature_name in enumerate(california_housing.feature_names):
    label = ttk.Label(root, text=f"{feature_name}:")
    label.grid(row=i, column=0, padx=10, pady=5)
    entry = ttk.Entry(root)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entry_widgets.append(entry)

# Buttons for predictions
linear_result = tk.StringVar()
nn_result = tk.StringVar()

linear_button = ttk.Button(root, text="Linear Regression Predict", command=predict_linear_regression)
linear_button.grid(row=len(california_housing.feature_names), column=0, columnspan=2, pady=10)
linear_result_label = ttk.Label(root, textvariable=linear_result)
linear_result_label.grid(row=len(california_housing.feature_names) + 1, column=0, columnspan=2, pady=10)

nn_button = ttk.Button(root, text="Neural Network Predict", command=predict_neural_network)
nn_button.grid(row=len(california_housing.feature_names) + 2, column=0, columnspan=2, pady=10)
nn_result_label = ttk.Label(root, textvariable=nn_result)
nn_result_label.grid(row=len(california_housing.feature_names) + 3, column=0, columnspan=2, pady=10)

# Run the GUI
root.mainloop()
