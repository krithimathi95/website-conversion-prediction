# website-conversion-prediction
Website Conversion Prediction is a machine learning project that predicts whether a website visitor is likely to convert based on their browsing and shopping behavior.
# Website Conversion Prediction Using Scikit-learn

## Project Description

Website Conversion Prediction is a machine learning project that predicts whether a website visitor is likely to convert based on their browsing and shopping behavior.

The project uses a **Random Forest Classifier** to analyze visitor session information such as session duration, pages viewed, previous visits, visitor type, cart activity, traffic source, and device type.

The project also includes a **Streamlit web application** that allows users to enter visitor session details and receive a conversion prediction along with the predicted conversion probability.

## Objectives

* Predict whether a website visitor will convert.
* Analyze visitor browsing behavior.
* Apply machine learning for website conversion prediction.
* Train a Random Forest classification model.
* Evaluate the trained model.
* Identify important features influencing conversion.
* Provide an interactive prediction interface using Streamlit.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Matplotlib
* Streamlit

## Machine Learning Algorithm

### Random Forest Classifier

The project uses the **Random Forest Classifier**, an ensemble machine learning algorithm that combines multiple decision trees to make predictions.

Random Forest is used for this project because the target variable is binary:

* `0` - Visitor does not convert
* `1` - Visitor converts

The model is configured with:

* 200 decision trees
* Maximum tree depth of 10
* Balanced class weights
* Random state of 42

## Dataset

The project uses a synthetic website conversion dataset stored in:

```text
data/website_conversion.csv
```

The dataset contains:

* 49 visitor sessions
* 7 input features
* 1 target variable

### Dataset Features

| Feature                  | Description                                              |
| ------------------------ | -------------------------------------------------------- |
| session_duration_seconds | Duration of the visitor's session in seconds             |
| pages_viewed             | Number of pages viewed during the session                |
| previous_visits          | Number of previous visits to the website                 |
| is_returning_visitor     | Indicates whether the visitor is a returning visitor     |
| cart_items               | Number of items added to the shopping cart               |
| traffic_source           | Source through which the visitor reached the website     |
| device_type              | Device used by the visitor                               |
| converted                | Target variable indicating whether the visitor converted |

### Target Variable

```text
0 = Will Not Convert
1 = Will Convert
```

The dataset contains 27 converted sessions and 22 non-converted sessions.

## Machine Learning Workflow

```text
Website Visitor Dataset
          |
          v
     Data Loading
          |
          v
   Feature Selection
          |
          v
     Train/Test Split
          |
          v
  Random Forest Classifier
          |
          v
       Prediction
          |
          v
     Model Evaluation
          |
          v
 Feature Importance Analysis
          |
          v
   Conversion Prediction
```

## Data Preparation

The project loads the dataset using Pandas and separates the input features from the target variable.

The selected input features are:

```text
session_duration_seconds
pages_viewed
previous_visits
is_returning_visitor
cart_items
traffic_source
device_type
```

The target variable is:

```text
converted
```

The dataset is divided into training and testing sets using an 80:20 split.

Stratified sampling is used to maintain the distribution of the target classes.

## Model Training

The Random Forest model is trained using the training dataset.

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)
```

After training, the model predicts conversion outcomes for the test dataset.

The trained model is saved as:

```text
models/website_conversion_model.pkl
```

## Model Evaluation

The project evaluates the model using:

* Accuracy Score
* Classification Report
* Confusion Matrix

The training script automatically displays these evaluation results when the model is trained.

## Feature Importance

The project calculates the importance of each input feature using the Random Forest model.

A feature importance chart is generated and saved as:

```text
outputs/feature_importance.png
```

This helps identify which visitor behavior characteristics contribute most to the model's conversion predictions.

## Streamlit Application

The project includes an interactive Streamlit application in:

```text
app.py
```

The application provides three main sections:

### 1. Live Visitor Session Evaluator

Users can enter visitor information such as:

* Session duration
* Pages viewed
* Previous visits
* Visitor type
* Cart items
* Traffic source
* Device type

The application then displays:

* Conversion prediction
* Conversion probability
* Drop-off likelihood
* Intent score
* CRO recommendations

### 2. Feature Importance

Displays the feature importance visualization generated during model training.

### 3. Historical Visitor Sessions

Displays the dataset and provides basic information such as:

* Total sessions
* Baseline conversion rate
* Average session duration

## Project Structure

```text
Website_Conversion_Prediction_Sklearn/
│
├── data/
│   └── website_conversion.csv
│
├── models/
│   └── website_conversion_model.pkl
│
├── outputs/
│   └── feature_importance.png
│
├── app.py
├── train_model.py
├── predict.py
├── requirements.txt
├── HOW_TO_RUN.md
├── run_frontend.bat
├── .gitignore
└── README.md
```

## Installation

### Step 1: Clone the Repository

```bash
git clone <your-github-repository-url>
```

### Step 2: Open the Project Directory

```bash
cd Website_Conversion_Prediction_Sklearn
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Train the Model

Run the following command:

```bash
python train_model.py
```

This will:

1. Load the website conversion dataset.
2. Select the required features.
3. Split the dataset into training and testing sets.
4. Train the Random Forest Classifier.
5. Evaluate the model.
6. Save the trained model.
7. Generate the feature importance chart.

## Make a Prediction

Run:

```bash
python predict.py
```

The prediction script uses a sample visitor session and displays whether the visitor is predicted to convert along with the conversion probability.

## Run the Streamlit Application

To start the interactive web application, run:

```bash
streamlit run app.py
```

The Streamlit application allows users to enter visitor information and receive real-time conversion predictions.

## Example Input

```text
Session Duration: 420 seconds
Pages Viewed: 8
Previous Visits: 3
Returning Visitor: Yes
Cart Items: 2
Traffic Source: Organic Search
Device Type: Desktop
```

The trained model uses these values to predict the visitor's conversion outcome and probability.

## Applications

This project can be used for:

* E-commerce conversion analysis
* Customer behavior analysis
* Website optimization
* Conversion Rate Optimization (CRO)
* Customer segmentation
* Marketing analytics
* Identifying potential cart abandonment
* Understanding website visitor behavior

## Future Improvements

The project can be further improved by:

* Using a larger real-world dataset.
* Comparing Random Forest with Logistic Regression, Decision Tree, XGBoost, and other algorithms.
* Applying cross-validation.
* Performing hyperparameter tuning.
* Adding ROC-AUC evaluation.
* Adding more detailed data visualizations.
* Implementing a database for storing visitor sessions.
* Deploying the Streamlit application online.
* Adding user authentication and session tracking.
* Integrating real-time website analytics data.

## Disclaimer

This project is intended for educational and demonstration purposes. Predictions depend on the quality and characteristics of the available dataset and should not be treated as guaranteed outcomes.
