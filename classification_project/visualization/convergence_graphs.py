import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from classification_project.visualization.visualization import Visualization

# Generate random data for the example
np.random.seed(42)
X = np.random.rand(1000, 32, 32, 3)
y = np.random.randint(2, size=1000)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a placeholder CNN model for the example
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model (Note: Replace this with your actual training process)
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))


# Example usage:
Visualization.plot_learning_curve(history)
Visualization.plot_roc_curve(model, X_val, y_val)
Visualization.plot_precision_recall_curve(model, X_val, y_val)

accuracy, loss, f1 = Visualization.calculate_validation_metrics(model, X_val, y_val)
Visualization.plot_convergence_graphs(history)
print("Validation Accuracy:", accuracy)
print("Validation Loss:", loss)
print("F1 Score:", f1)
