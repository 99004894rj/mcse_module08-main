import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

# Ensure TensorFlow is using the GPU with ROCm
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Configured TensorFlow to use GPU:", gpus)
    except RuntimeError as e:
        print("Failed to set memory growth:", e)

# Load the dataset efficiently
file_path = r'D:\IOT\conn.log.labelled.txt'
print("Loading dataset...")
data = pd.read_csv(file_path, sep='\t', comment='#', header=None, low_memory=False, dtype=str)

# Define and print column names
columns = ['ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'proto', 'service', 'duration',
           'orig_bytes', 'resp_bytes', 'conn_state', 'local_orig', 'local_resp', 'missed_bytes', 'history',
           'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes', 'tunnel_parents']
data.columns = columns
print("Columns in the dataset:", data.columns)

# Extract features and labels, convert text data to numerical features
print("Extracting and transforming data...")
X_text = data.drop(columns=['label', 'detailed-label'], errors='ignore').astype(str)
y = data['label'] if 'label' in data.columns else np.zeros(data.shape[0])
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X_text.apply(lambda x: ' '.join(x), axis=1))

# Scale and encode features
scaler = StandardScaler(with_mean=False)
X = scaler.fit_transform(X)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset and define the neural network architecture
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_dim=X_train.shape[1]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
