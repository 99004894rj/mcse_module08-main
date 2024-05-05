import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Check if GPU is available and set TensorFlow to use GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Step 1: Load the dataset
file_path = r'C:\Users\rodyj\Documents\data\conn.log.labelled.txt'
print("Loading dataset...")
try:
    data = pd.read_csv(file_path, sep='\t', skiprows=6, low_memory=False, dtype=str)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    exit(1)

# Print column names to verify the dataset
print("Columns in the dataset:")
print(data.columns)

# Extracting features
print("Extracting features...")
X_text = data.drop(columns=['label', 'detailed-label']).values
y = data['label'].values

# Convert text data to numerical features using CountVectorizer
print("Converting text data to numerical features...")
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X_text).toarray()

# Scale numerical features
print("Scaling numerical features...")
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encode labels
print("Encoding labels...")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Step 3: Split the dataset into training and testing sets
print("Splitting the dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the number of features you want to select
k = 10

# Define your preprocessing and modeling steps
print("Defining preprocessing and modeling steps...")
pipeline = Pipeline([
    ('feature_selector', SelectKBest(score_func=f_classif, k=k)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
print("Training the model...")
pipeline.fit(X_train, y_train)

# Select top k features for training and testing data
print("Selecting top k features for training and testing data...")
X_train_selected = pipeline.named_steps['feature_selector'].transform(X_train)
X_test_selected = pipeline.named_steps['feature_selector'].transform(X_test)

# Step 4: Define the neural network architecture
print("Defining the neural network architecture...")
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_selected.shape[1],)),
    Dense(64, activation='relu'),
    Dense(2, activation='sigmoid')
])

# Step 5: Define the loss function and optimizer
print("Defining the loss function and optimizer...")
model.compile(optimizer=Adam(lr=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Train the model
print("Training the model...")
model.fit(X_train_selected, y_train, epochs=10, batch_size=32)

# Step 7: Evaluate the model
print("Evaluating the model...")
loss, accuracy = model.evaluate(X_test_selected, y_test)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')
