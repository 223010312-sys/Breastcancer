# Breast Cancer Classification Project

This repository contains a machine learning project focused on **Breast Cancer diagnosis prediction** using Python and scikit-learn. The analysis is performed in a Jupyter Notebook (`Breastcancer.ipynb`).

---

## 📌 Project Overview

The goal of this project is to build a predictive model that classifies breast tumors as **benign** or **malignant** based on various cell nucleus measurements.

The workflow consists of:

* Data loading and preprocessing
* Handling categorical labels using Label Encoding
* Exploratory data inspection
* Dimensionality reduction and clustering using **K-Means**
* Training a **K-Nearest Neighbors (KNN)** classification model
* Model evaluation

---

## 📁 Dataset

The dataset used in this project is `Dataset.csv`, which contains features such as:

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* And many more diagnostic features

The dataset also includes a `diagnosis` column indicating:

* **M** → Malignant
* **B** → Benign

This column is encoded into numerical form using scikit-learn's `LabelEncoder`.

---

## ⚙️ Technologies Used

* **Python 3**
* **Pandas** – data manipulation
* **NumPy** – numerical computing
* **Matplotlib** – data visualization
* **scikit-learn** – machine learning models (LabelEncoder, KMeans, KNN)
* **Jupyter Notebook** – interactive experimentation

---

## 🔧 Steps Performed

### 1. **Data Preprocessing**

* Loaded the dataset into a Pandas DataFrame
* Removed unnecessary columns (`id`, `Unnamed: 32`)
* Encoded the target variable `diagnosis`

### 2. **Clustering with K-Means**

* Applied K-Means to group data into cluster labels
* Added cluster labels (`cl`) to both training and testing sets

### 3. **Model Building with KNN**

* Trained a KNN classifier using the training data
* Predicted outcomes on the test set

---

## ▶️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/yourrepo.git
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Open the notebook:

```bash
jupyter notebook Breastcancer.ipynb
```

---

## 📊 Results

The notebook demonstrates the model's prediction process and includes intermediate outputs such as:

* Cluster assignments
* Model predictions
* Dataset info before and after preprocessing

You can modify hyperparameters like `n_neighbors` in the KNN model to further optimize performance.

---

## ✨ Future Improvements

* Add accuracy, precision, recall, and confusion matrix
* Visualize clusters using PCA/TSNE
* Try more advanced models (Random Forest, SVM, XGBoost)
* Implement a proper train-test split and cross-validation

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 🤝 Contributions

Contributions, issues, and pull requests are welcome!

---

## 🙌 Acknowledgements

Dataset sourced from typical Breast Cancer Wisconsin diagnostic datasets and processed using scikit-learn tools.
