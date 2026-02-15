# Fraud Detection System

# Project Background

This project aims to train a model to detect fraudulent transaction in credit card transactions. This is built as a learning project to explore end-to-end ML workflows, from data prep and training to prediction and visualization. A use-case for this project is that it can be adapted to analyze fraud likelihood and produce insights/visuals that help interpret how much of a transaction is predicted to be fraudulent.

In this project, I built a pipeline that:

- Prepares and preprocesses data (cleaning, splitting, scaling)

- Trains classification models (logistic regression and random forest)

- Evaluates performance (ROC-AUC, PR-AUC, confusion matrix, classification report)

- Generates predictions (via the saved pipeline)

# Features

- Modular end-to-end ML pipeline (preprocess -> train -> evaluate -> predict)

- Supports multiple models (baseline logistic regression + tree-based random forest)

- Clear script separation for preprocessing and training step

# Script & Module Descriptions

## **src/preprocessing.py** 

**Purpose**: Handles data cleaning and preprocessing.

**Key Functions:**

**- basic_clean(df):** Removes duplicates and fills missing numeric values with column medians.

**- split_x_y(df):** Separates features (x) from target (y, column "Class").

**- build_preprocessor(x)**: Builds a "ColumnTransformer":

- Scales numeric columns Time and Amount

- Passes other features through unchanged

Role in Pipeline: Ensures data is clean and ready for modeling. First step before training.

## **src/train.py**

**Purpose:** Trains a classification model, evaluates performance, and saves results.

**Key Features:**

**Model options:**

**- Logistic Regression (logreg):** supports class_weight="balanced".

**- Random Forest (rf):** 400 trees, with class_weight support.

**Handles class imbalance:**

- Default class_weight="balanced"

**Preprocessing pipeline:**

- Scales numeric features Time and Amount

- Leaves other features unchanged

- Integrated into a Pipeline or ImbPipeline if SMOTE is used

**Train/test split:**

- Stratified to preserve fraud ratio

- Configurable test_size (default 0.2)

**Evaluation metrics:**

- ROC-AUC, PR-AUC

- Confusion matrix

- Full classification report

- Dataset statistics (number of training/testing samples, fraud rates)

**Artifact creation:**

**pipeline.pkl:** pickled preprocessing + model pipeline

**metrics.json:** JSON summary of evaluation metrics and training configuration

## artifacts/

**Stores output from training:**

**- pipeline.pkl:** Trained model pipeline

**- metrics.json:** Evaluation metrics (ROC-AUC, PR-AUC, confusion matrix, classification report, etc.)

# Limitaitons



# Reflections








