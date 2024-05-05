IoT Network Traffic Classification

This project focuses on classifying IoT network traffic into benign and malicious categories using machine learning techniques, specifically a combination of feature selection, preprocessing, and deep learning.
Dataset

The dataset used in this project is con.log.labelled, which contains labeled IoT network traffic data. The dataset is expected to be in CSV format and should have two columns: feature containing the textual features of network traffic and label containing the corresponding labels (benign or malicious).
Requirements

    Python 3
    Libraries:
        pandas
        scikit-learn
        TensorFlow
        numpy

Installation

    Clone this repository:

    bash

git clone https://github.com/99004894rj/mcse_module08.git

Navigate to the project directory:

bash

cd mcse_module08

Install the required libraries:

bash

    pip install -r requirements.txt

Usage

    Place your dataset file (con.log.labelled) in the project directory.

    Run the cnn.py script:

    bash

    python cnn.py

    This will train a Convolutional Neural Network (CNN) model on the dataset and evaluate its performance.

License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.