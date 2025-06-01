# Sentiment Analysis on Amazon Product Reviews

## 📌 Project Overview

This project focuses on performing sentiment analysis on customer product reviews from Amazon. The main objective is to classify reviews as *positive* or *negative* using machine learning techniques. The notebook contains steps from data preprocessing to model evaluation.

The analysis was done using Python on Google Colab, leveraging Natural Language Processing (NLP) methods and machine learning models to understand customer sentiments.

## 📂 Raw Dataset

The dataset used in this project comes from Kaggle and contains product reviews with corresponding star ratings.

🔗 [Amazon Product Reviews Dataset on Kaggle](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products)

## 📊 Insight & Findings

- Majority of the reviews in the dataset are *positive* (ratings 4–5).
- Reviews with ratings 1–2 are considered *negative*.
- After preprocessing and converting reviews into numerical features using *TF-IDF*, we trained several models.
- *Logistic Regression* showed the best performance with an *accuracy above 85%*.
- Key preprocessing steps included:
  - Removing stopwords
  - Lowercasing
  - Tokenization
  - TF-IDF vectorization

## 🤖 AI Support Explanation

This project integrates AI through the use of:
- *Natural Language Processing (NLP):* to process and clean raw review text.
- *Machine Learning (Scikit-learn):* for training classifiers (Logistic Regression, Naive Bayes).
- *AI-assisted environments:* such as Google Colab, which allows accelerated training and testing.

AI techniques helped automate the sentiment labeling process, enabling scalable analysis of thousands of product reviews efficiently.

---

📁 For more details, check the full notebook in this repository.
