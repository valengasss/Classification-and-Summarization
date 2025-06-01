# Sentiment Analysis of Shopee COD Reviews

## 📌 Project Overview

This project focuses on performing sentiment analysis on *Shopee Cash on Delivery (COD)* customer reviews. The main goal is to classify the sentiment of user reviews into *positive* and *negative* categories using Natural Language Processing (NLP) and machine learning.

Key steps in the notebook include:
- Importing and exploring the dataset
- Cleaning and preprocessing review texts
- Converting text to numerical features using TF-IDF
- Training a Logistic Regression model
- Evaluating model performance using classification report and confusion matrix

The entire analysis was conducted on *Google Colab* using Python and popular ML libraries such as Scikit-learn and NLTK.


## 📂 Raw Dataset

The dataset used contains Shopee product reviews, specifically focused on COD transactions

🔗 [https://www.kaggle.com/datasets/alvianardiansyah/dataset-ulasan-pengguna-shopee]

## 📊 Insight & Findings

- *Sentiment Distribution:* Reviews with ratings 4–5 were labeled as *positive, while 1–2 were labeled **negative*. Neutral ratings (3) were excluded.
- *Data Preprocessing:* Reviews were cleaned by removing special characters, converting to lowercase, and eliminating stopwords.
- *Model Performance:* Logistic Regression achieved an accuracy of *~89%*, indicating good sentiment classification performance.
- *Word Frequency Analysis:* Frequently used positive and negative words were identified to understand common customer expressions

## 🤖 AI Support Explanation

The AI capabilities applied in this project include:
- *Natural Language Processing (NLP):* For text preprocessing, cleaning, and vectorization.
- *TF-IDF Vectorization:* To numerically represent textual data for machine learning input.
- *Machine Learning (Logistic Regression):* To classify reviews into sentiment categories.
- *Google Colab as AI environment:* For scalable and cloud-based computation.


📁 For more details, check the full notebook in this repository.
