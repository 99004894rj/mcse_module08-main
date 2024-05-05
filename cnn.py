import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier  # Added RandomForestClassifier
from tensorflow.keras import layers, models

# Step 1: Load the dataset
file_path = r'D:\IOT\con.log.labelled'
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")

# Extracting features
X_text = data['history'].values  # Assuming 'history' column contains text data
y = data['label'].values  # Labels

# Convert text data to numerical features using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X_text)

# Scale numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X.toarray())

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the number of features you want to select
k = 10

# Define your preprocessing and modeling steps
pipeline = Pipeline([
    ('feature_selector', SelectKBest(score_func=f_classif, k=k)),  # Select top k features based on ANOVA F-value
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  # RandomForest classifier
])

# Train the model
pipeline.fit(X_train, y_train)

# Select top k features for training and testing data
X_train_selected = pipeline.named_steps['feature_selector'].transform(X_train)
X_test_selected = pipeline.named_steps['feature_selector'].transform(X_test)

# Step 4: Define the CNN architecture
model = models.Sequential([
    layers.Input(shape=(X_train_selected.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Step 5: Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Step 6: Train the model
model.fit(X_train_selected, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Step 7: Evaluate the model
loss, accuracy = model.evaluate(X_test_selected, y_test)
print(f'Test Accuracy: {accuracy}')
