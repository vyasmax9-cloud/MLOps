# for data manipulation
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score

# for model serialization
import joblib
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

api = HfApi()

Xtrain_path = "vyasmax9/tourism-prediction/Xtrain.csv"
Xtest_path = "vyasmax9/tourism-prediction/Xtest.csv"
ytrain_path = "vyasmax9/tourism-prediction/ytrain.csv"
ytest_path = "vyasmax9/tourism-prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# Define features
numeric_features =  ['Age', 'NumberOfPersonVisiting', 'NumberOfFollowups', 'DurationOfPitch', 'PitchSatisfactionScore']
categorical_features = ['CustomerID', 'TypeofContact', 'Occupation', 'Gender', 'CityTier', 'MaritalStatus', 'PreferredPropertyStar', 'Designation']

# Set the class weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
class_weight

# Preprocessing pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100, 125, 150],   # number of tree to build
    'xgbclassifier__max_depth': [2, 3, 4],     # maximum depth of each tree
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],  # learning rate
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],   # percentage of attributes to be considered (randomly) for each tree
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],   # percentage of attributes to be considered (randomly) for each level of tree
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],      # L2 regularization factor
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Hyperparameter tuning with GridSearchCV 
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1
)
grid_search.fit(Xtrain, ytrain)

# Best model
best_model = grid_search.best_estimator_
best_model

# Set the classification thresold
classification_thresold = 0.45

# Make Predictions on the training data
y_pred_train_proba = best_model.predict_proba(Xtrain) [:, 1]       
y_pred_train = (y_pred_train_proba >= classification_thresold).astype(int)

# Make Predictions on the test data
y_pred_test_proba = best_model.predict_proba(Xtest) [:, 1]       
y_pred_test = (y_pred_test_proba >= classification_thresold).astype(int)


# Generate a classification report to evaluate model performance on training set
print(classification_report(ytrain, y_pred_train))

# Generate a classification report to evaluate model performance on test set
print(classification_report(ytest, y_pred_test))


# Save best model
joblib.dump(best_model, "tourism_prediction_model.joblib") 


# Upload to Hugging Face
repo_id = "vyasmax9/tourism-prediction"
repo_type = "model"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Model Space '{repo_id}' created.")

api.upload_file(
    path_or_fileobj="tourism_predictio_model.joblib", 
    path_in_repo="tourism_prediction_model.joblib", 
    repo_id=repo_id,
    repo_type=repo_type,
)

