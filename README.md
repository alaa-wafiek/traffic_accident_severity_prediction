# traffic_accident_severity_prediction
A comparative study of Classical Models (XGBoost, RF) vs. Neural Networks (Weighted, ResNet, Focal Loss) for accident severity prediction.

# 🚦 Traffic Accident Severity Classification

## 📌 Project Overview
This project aims to predict the severity of traffic accidents (**Minor, Moderate, or Severe**) based on environmental conditions, temporal data, and road features. 

The core challenge addressed in this project is **Class Imbalance** (severe accidents are rare). We tackled this by benchmarking Classical Machine Learning models against custom **Deep Learning architectures**, including **ResNet** with **Focal Loss** and **Weighted Neural Networks**.

## 🚀 Key Features
* **End-to-End Pipeline:** From raw data cleaning to model deployment simulation.
* **Advanced Preprocessing:** * Handling missing values and outliers.
    * Feature Engineering (Rush Hour, Weekend flags, Temporal extraction).
    * Leakage prevention by removing post-crash variables.
* **Model Comparison:**
    * **Classical:** Logistic Regression, Random Forest, XGBoost.
    * **Deep Learning:** Custom ANN, Weighted ANN, SMOTE-based ANN, and ResNet.
* **Imbalance Handling:** Implementation of Class Weights, Focal Loss, and Oversampling.
* **Optimization:** Hyperparameter tuning using Adam/SGD/RMSprop and Xavier Initialization.

## 📂 Dataset
The dataset includes traffic accident records with features such as:
* **Environmental:** Weather, Lighting, Road Surface Condition.
* **Temporal:** Crash Date, Time of Day.
* **Configuration:** Speed Limit, Traffic Control Device, First Crash Type.

*> Note: The target variable `most_severe_injury` was mapped to 3 classes: Minor (0), Moderate (1), and Severe (2).*

##  Technologies Used
* **Languages:** Python
* **Libraries:** * `Pandas`, `NumPy` (Data Manipulation)
    * `Matplotlib`, `Seaborn` (Visualization)
    * `Scikit-Learn` (Preprocessing & Classical ML)
    * `TensorFlow / Keras` (Deep Learning)
    * `XGBoost` (Gradient Boosting)

##  Methodology

### 1. Exploratory Data Analysis (EDA)
We analyzed correlations and distributions, visualizing:
* Accident spikes during Rush Hours.
* Impact of Lighting and Weather on crash severity.
* Interaction between Crash Type and Injury Severity.

### 2. Model Architecture
We implemented multiple neural network variants to improve performance on the minority class:
* **Baseline NN:** Standard Dense layers with Dropout and BatchNormalization.
* **Weighted NN:** Applied `class_weight` to penalize misclassification of severe accidents.
* **ResNet (Residual Network):** Utilized Skip Connections (`Add` layers) to allow for deeper architectures without vanishing gradients.
* **Custom Loss:** Implemented **Focal Loss** to focus learning on hard-to-classify examples.

### 3. Hyperparameter Tuning
Experiments were conducted using a custom configuration loop:
* **Optimizers:** Adam (Best Performing), SGD, RMSprop.
* **Initialization:** Xavier (Glorot) Initialization.
* **Regularization:** Dropout (0.3 - 0.5), Early Stopping, and Learning Rate Decay (`ReduceLROnPlateau`).

## 📈 Results
| Model | Strategy | Key Strength |
| :--- | :--- | :--- |
| **Random Forest** | Ensemble | Robust baseline, good general accuracy. |
| **XGBoost** | Gradient Boosting | Excellent on tabular data, high precision. |
| **Weighted ResNet** | Deep Learning | **Best Recall for Severe Cases** (Successfully handled imbalance). |

*> Detailed confusion matrices and loss curves can be found in the notebook.*

## ⚙️ How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/alaa-wafiek/traffic-accident-severity.git](https://github.com/alaa-wafiek/traffic-accident-severity.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn tensorflow xgboost
    ```
3.  **Run the Notebook:**
    Open `datascienceproject_final.ipynb` in Jupyter Notebook or Google Colab.


## 📜 License
This project is for educational purposes as part of the Intro to Data Science course.
