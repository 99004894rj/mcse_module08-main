import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix

# Check if GPU is available and set PyTorch to use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load the dataset
file_path = '/path/to/your/dataset.txt'  # Update your path
print("Loading dataset...")
data = pd.read_csv(file_path, sep='\t', comment='#', header=None, low_memory=False, dtype=str, nrows=250000)

# Define column names
columns = ['ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'proto', 'service', 'duration',
           'orig_bytes', 'resp_bytes', 'conn_state', 'local_orig', 'local_resp', 'missed_bytes', 'history',
           'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes', 'tunnel_parents', 'label']
data.columns = columns
print("Columns in the dataset:")
print(data.columns)

# Extracting features and labels
print("Extracting features and labels...")
X_text = data.drop(columns=['label'], errors='ignore').astype(str)
y = data['label']

# Convert text data to numerical features using CountVectorizer
print("Converting text data to numerical features...")
vectorizer = CountVectorizer(min_df=0.01, max_df=0.95, max_features=10000)
X = vectorizer.fit_transform(X_text.apply(lambda x: ' '.join(x), axis=1))

# Convert to PyTorch sparse tensor
print("Converting to PyTorch sparse tensor...")
X_coo = X.tocoo()
values = torch.tensor(X_coo.data, dtype=torch.float32)
indices = torch.tensor(np.vstack((X_coo.row, X_coo.col)), dtype=torch.long)
shape = torch.Size(X_coo.shape)
X_tensor = torch.sparse_coo_tensor(indices, values, shape).to(device)

# Encode labels
print("Encoding labels...")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y_tensor = torch.tensor(y, dtype=torch.long).to(device)
print("Unique labels:", np.unique(y_tensor.cpu().numpy()))

# Split the dataset
print("Splitting the dataset into training and testing sets...")
indices = np.arange(X_tensor.shape[0])
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
X_train = X_tensor.index_select(0, torch.tensor(train_indices, dtype=torch.long).to(device))
y_train = y_tensor.index_select(0, torch.tensor(train_indices, dtype=torch.long).to(device))
X_test = X_tensor.index_select(0, torch.tensor(test_indices, dtype=torch.long).to(device))
y_test = y_tensor.index_select(0, torch.tensor(test_indices, dtype=torch.long).to(device))

# Custom collate function for handling sparse tensors
def custom_collate(batch):
    data = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return torch.stack(data), torch.tensor(targets)

# Create TensorDatasets and DataLoaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

# Define the neural network architecture
print("Defining the neural network architecture...")
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, len(np.unique(y)))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNN().to(device)

# Define the loss function and optimizer
print("Defining the loss function and optimizer...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Check initial predictions without training
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    sample_data = X_train[:10]  # Take a sample batch from the training set
    sample_labels = y_train[:10]
    outputs = model(sample_data)
    print("Sample outputs:", outputs)
    print("Sample labels:", sample_labels)

# Check the loss for initial predictions
loss = criterion(outputs, sample_labels)
print("Initial sample loss:", loss.item())

# Train the model
print("Training the model...")
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# Evaluate the model
print("Evaluating the model...")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy:.4f}')

# Save the model
model_save_path = '/path/to/your/model.pth'  # Update your path
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
