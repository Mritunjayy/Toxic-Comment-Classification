# Toxic-Comment-Classification
This repository contains code for a multi-label text classification project using TensorFlow, specifically designed for the Toxic Comment Classification. The project aims to build a machine learning model capable of identifying different types of toxic comments in text data.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## We have broken down a large dataset into small chunks of 6 new datasets using 
from google.colab import drive
drive.mount('/content/drive')
!apt-get install git
!git config --global user.name "Mritunjayy"
!git config --global user.email "pmritunjay635@gmail.com"
import os
os.environ['GITHUB_TOKEN'] = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
!git clone https://{os.environ['GITHUB_TOKEN']}@github.com/Mritunjayy/Toxic-Comment-Classification.git
%cd Toxic-Comment-Classification
import pandas as pd
file_path = '/content/train.csv'
df = pd.read_csv(file_path)
chunk_size = 25000  # Adjust based on your needs
for i, chunk in enumerate(range(0, df.shape[0], chunk_size)):
    chunk_path = f'train_{i}.csv'
    df.iloc[chunk:chunk + chunk_size].to_csv(chunk_path, index=False)
    !git add {chunk_path}
    !git commit -m "Add chunk {i} of large CSV file"
!git push origin main
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Multi-label text classification is a complex task where each input text can belong to multiple categories simultaneously. This project utilizes TensorFlow and deep learning techniques to effectively classify comments into one or more toxicity categories such as toxic, severe toxic, obscene, threat, insult, and identity hate.

# One-vs-Rest (OvR)
Description:
One-vs-Rest, also known as One-vs-All (OvA), is a strategy where a separate binary classifier is trained for each label. For each classifier, the samples are divided into two classes: one where the given label is present (positive class) and one where it is absent (negative class). During prediction, each classifier independently predicts the presence or absence of its respective label.

# Binary Relevance (BR)
Description:
Binary Relevance is a simple method where each label is treated as a separate and independent binary classification problem. Similar to One-vs-Rest, it trains one binary classifier per label without considering the interdependencies between labels.

# Classifier Chains (CC)
Description:
Classifier Chains is a more advanced method that considers label dependencies by arranging the labels in a chain and training a sequence of binary classifiers. Each classifier in the chain is trained to predict one label, using not only the input features but also the predictions of all previous classifiers in the chain as additional features.

# MultiOutput Classifier
Description:
MultiOutput Classifier is a meta-estimator that fits a separate classifier for each output variable (label). It is essentially an extension of multi-label classification to multi-output regression or classification. The approach can use any base classifier to handle each label independently but in a unified framework.
