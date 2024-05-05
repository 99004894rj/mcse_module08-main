import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

# Check if GPU is available and set TensorFlow to use GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Step 1: Load the dataset
file_path = r'D:\IOT\conn.log.labelled.txt'
print("Loading dataset...")
data = pd.read_csv(file_path, sep='\t', comment='#', header=None, low_memory=False, dtype=str)

# Define column names
columns = ['ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'proto', 'service', 'duration',
           'orig_bytes', 'resp_bytes', 'conn_state', 'local_orig', 'local_resp', 'missed_bytes', 'history',
           'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes', 'tunnel_parents']
data.columns = columns

# Print column names to verify the dataset structure
print("Columns in the dataset:")
print(data.columns)

# Extracting features and labels
print("Extracting features...")
X_text = data.drop(columns=['label', 'detailed-label'], errors='ignore').astype(str)
y = data['label'] if 'label' in data.columns else np.zeros(data.shape[0])  # Creating dummy labels if not available

# Convert text data to numerical features using CountVectorizer
print("Converting text data to numerical features...")
vectorizer = CountVectorizer()
# Ensure X_text is correctly transformed, each row corresponds to a document.
X = vectorizer.fit_transform(X_text.apply(lambda x: ' '.join(x), axis=1))

# Check the shapes to ensure consistency
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {len(y)}")

# Scale numerical features
print("Scaling numerical features...")
scaler = StandardScaler(with_mean=False)  # Use with_mean=False to support sparse matrices
X = scaler.fit_transform(X)

# Encode labels
print("Encoding labels...")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Check shapes again after encoding
print(f"Shape of X after processing: {X.shape}")
print(f"Shape of y after encoding: {len(y)}")

# Step 3: Split the dataset into training and testing sets
print("Splitting the dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Define the neural network architecture using TensorFlow
print("Defining the neural network architecture...")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_dim=X_train.shape[1]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
])

# Step 5: Define the loss function and optimizer
print("Defining the loss function and optimizer...")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Train the model
print("Training the model...")
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Step 7: Evaluate the model
print("Evaluating the model...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
