import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Set environment variable to avoid "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Assuming CUDA is available, if not, this will fall back to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Step 1: Load the dataset
file_path = r'C:\Users\rodyj\Documents\data\conn.log.labelled.txt'
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    exit(1)  # Exit the script if file not found

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

# Convert NumPy arrays to PyTorch tensors
X_train_selected = torch.tensor(X_train_selected, dtype=torch.float32)
X_test_selected = torch.tensor(X_test_selected, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Step 4: Define the neural network architecture
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Output size is 2 for binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Initialize the model
input_size = X_train_selected.shape[1]
model = NeuralNet(input_size)

# Step 5: Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 6: Train the model
num_epochs = 10
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_selected)
    loss = criterion(outputs, y_train)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 7: Evaluate the model
with torch.no_grad():
    outputs = model(X_test_selected)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print(f'Test Accuracy: {accuracy:.4f}')
