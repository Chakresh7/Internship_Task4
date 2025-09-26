# Breast Cancer Classification using Logistic Regression

This repository contains a Jupyter Notebook (**Internship_task_4.ipynb**) implementing a machine learning pipeline to classify breast cancer tumors as **Malignant (M)** or **Benign (B)** using the Breast Cancer Wisconsin dataset.

## 📂 Project Structure

* `Internship_task_4.ipynb` — Notebook with the full code and explanations
* `data.csv` — Dataset used for training and testing

## 📝 Overview

The notebook demonstrates the following steps:

1. **Data Loading & Exploration**

   * Loaded `data.csv`
   * Checked shape, data types, and missing values using `df.head()`, `df.shape`, `df.info()`

2. **Data Preprocessing**

   * Dropped irrelevant column `Unnamed: 32`
   * Split features (`X`) and target (`y`) — target is the `diagnosis` column
   * Train-test split with 80% training and 20% testing data
   * Standardized features using `StandardScaler`

3. **Model Training**

   * Trained a Logistic Regression model on the standardized training data

4. **Model Evaluation**

   * Generated predictions on the test set
   * Evaluated using:

     * Confusion Matrix (plotted with Seaborn heatmap)
     * Accuracy Score
     * Classification Report (Precision, Recall, F1-score)
   * Achieved **97% accuracy**

5. **Threshold Tuning**

   * Used `predict_proba` to obtain class probabilities
   * Applied a custom threshold of **0.3** for the positive class (Malignant)
   * Compared performance metrics with the custom threshold

## 🛠️ Technologies Used

* Python 3
* NumPy
* Pandas
* Matplotlib & Seaborn (visualization)
* Scikit-learn (model building & evaluation)

## 🚀 How to Run

1. Clone the repository

   ```bash
   git clone https://github.com/yourusername/yourrepo.git
   cd yourrepo
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

   *(You can create a `requirements.txt` with: `numpy pandas matplotlib seaborn scikit-learn`)*

3. Run the Jupyter Notebook

   ```bash
   jupyter notebook Internship_task_4.ipynb
   ```

4. Upload the `data.csv` file or ensure it is in the same folder as the notebook.

## 📊 Results

* **Initial Logistic Regression Model Accuracy:** 97%
* **Custom Threshold (0.3) Accuracy:** 96%
* Precision/Recall trade-offs explored

## 🧠 Key Learnings

* Preprocessing real-world datasets for ML
* Training and evaluating a classification model
* Understanding & applying classification thresholds

## 📜 License

This project is open-source and available under the MIT License.
