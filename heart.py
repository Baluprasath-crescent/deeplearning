import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the Heart Disease Dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv"
data = pd.read_csv(url)

# Prepare features (X) and labels (y)
X = data.drop(columns="target").values  # Features
y = data["target"].values               # Labels (binary: 0 or 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
    Dense(8, activation='relu'),                                    # Hidden layer
    Dense(1, activation='sigmoid')                                  # Output layer (binary classification)
])

# Compile the model
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.2, verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Make predictions
predictions = (model.predict(X_test) > 0.5).astype("int32").flatten()

# Display predictions and true labels
print(f"Predicted classes: {predictions}")
print(f"True classes: {y_test}")
