import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load your dataset
merged_file = 'merged_FI_data.csv'
df = pd.read_csv(merged_file)

# Assuming the feature columns are all columns except 'label'
data = df.drop(columns=['label']).values
labels = df['label'].values

# Adjust labels to be zero-indexed
labels = labels - 1

# Normalize data
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Define window size
window_size = 10
num_features = data.shape[1]

# Function to create sequences
def create_sequences(data, labels, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])  # Keep the sequence as 2D for CNN
        y.append(labels[i + window_size])
    return np.array(X), np.array(y)

# Create sequences
X, y = create_sequences(data, labels, window_size)

# Reshape X for CNN [samples, height, width, channels]
X = X.reshape((X.shape[0], window_size, num_features, 1))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize and train CNN model
def build_cnn_model(window_size, num_features, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(window_size, num_features, 1)),
        MaxPooling2D((2, 2), padding='same'),  # Ensure pooling is valid
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),  # Ensure pooling is valid
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),  # Ensure pooling is valid
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

num_classes = len(np.unique(labels))
model = build_cnn_model(window_size, num_features, num_classes)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=250, batch_size=64, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

# Plot loss and accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Curve')

plt.show()

# Make predictions
y_pred = np.argmax(model.predict(X_test), axis=-1)

# Compute evaluation metrics
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print('Confusion Matrix:')
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(num_classes), yticklabels=np.arange(num_classes))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Visualize a sample sequence
def plot_sample_sequence(sequence, label):
    plt.figure(figsize=(10, 5))
    plt.imshow(sequence, aspect='auto', cmap='viridis')
    plt.title(f'Sample Sequence - Label: {label}')
    plt.colorbar()
    plt.xlabel('Feature Index')
    plt.ylabel('Time Step')
    plt.show()

# Pick a random sample from the test set
sample_idx = np.random.randint(0, len(X_test))
sample_sequence = X_test[sample_idx, :, :, 0]
sample_label = y_test[sample_idx]

plot_sample_sequence(sample_sequence, sample_label)
