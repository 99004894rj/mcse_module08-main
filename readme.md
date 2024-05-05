
# IoT Network Traffic Classification

This project focuses on classifying IoT network traffic into benign and malicious categories using machine learning techniques, including feature selection, preprocessing, and deep learning models implemented in both TensorFlow and PyTorch.

## Dataset

The dataset used in this project is `conn.log.labelled`, which contains labeled IoT network traffic data. The dataset is expected to be in CSV format and should include a variety of network traffic features with corresponding labels (benign or malicious).

## Requirements

- Python 3
- Libraries:
  - pandas
  - scikit-learn
  - TensorFlow
  - PyTorch
  - numpy

## Installation

Clone this repository:
```bash
git clone https://github.com/99004894rj/mcse_module08.git
```

Navigate to the project directory:
```bash
cd mcse_module08
```

Install the required libraries:
```bash
pip install -r requirements.txt
```

## Usage

Place your dataset file (`conn.log.labelled`) in the project directory.

To check CUDA availability on your machine, you can run the following script:
```bash
python check_cuda.py
```

To check the column consistency of your dataset, use:
```bash
python check_columns.py
```

### TensorFlow Model
Run the `cnn_tensorflow.py` script to train a Convolutional Neural Network (CNN) model using TensorFlow on the dataset and evaluate its performance:
```bash
python cnn_tensorflow.py
```

### PyTorch Model
Run the `cnn_pytorch.py` script to train a similar CNN model using PyTorch:
```bash
python cnn_pytorch.py
```

This will train the model on the dataset and evaluate its performance, printing the accuracy and loss metrics.

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.
