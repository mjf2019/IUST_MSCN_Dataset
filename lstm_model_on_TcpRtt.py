import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

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
window_size = 3  # Example window size

# Function to create sequences
def create_sequences(data, labels, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(labels[i + window_size])
    return np.array(X), np.array(y)

# Create sequences
X, y = create_sequences(data, labels, window_size)

# Ensure X has the correct shape [samples, time steps, features]
num_features = data.shape[1]  # Number of features
X = X.reshape((X.shape[0], X.shape[1], num_features))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build the LSTM model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(window_size, num_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.2))  # Add dropout to prevent overfitting
model.add(Dense(len(np.unique(labels)), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

# Make predictions
def predict_sequence(model, data, window_size):
    predictions = []
    for i in range(len(data) - window_size):
        sequence = data[i:i + window_size]
        sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
        prediction = model.predict(sequence)
        predictions.append(np.argmax(prediction))  # Choose class with highest probability
    return np.array(predictions)

# Predict on the test set
y_pred = predict_sequence(model, X_test, window_size)

# Compute evaluation metrics
precision = precision_score(y_test[window_size:], y_pred, average='weighted')
recall = recall_score(y_test[window_size:], y_pred, average='weighted')
f1 = f1_score(y_test[window_size:], y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test[window_size:], y_pred)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print('Confusion Matrix:')
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(len(np.unique(labels))), yticklabels=np.arange(len(np.unique(labels))))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
print('Classification Report:')
print(classification_report(y_test[window_size:], y_pred))
