# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Log in to Hugging Face (important for accessing datasets)
login(token=os.getenv("HF_TOKEN"))

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/vyasmax9/tourism-prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Define the target variable for the classification task
target = 'ProdTaken'

# List of numerical features in the dataset
numeric_features = ['Age', 'NumberOfPersonVisiting', 'NumberOfFollowups', 'DurationOfPitch', 'PitchSatisfactionScore']

# List of categorical features in the dataset
categorical_features = ['CustomerID', 'TypeofContact', 'Occupation', 'Gender', 'CityTier', 'MaritalStatus', 'PreferredPropertyStar', 'Designation']

# Define predictor matrix (X) using selected numeric and categorical features
X = df[numeric_features + categorical_features]

# Define target variable
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.to_csv("Xtrain.csv",index=False)
X_test.to_csv("Xtest.csv",index=False)
y_train.to_csv("ytrain.csv",index=False)
y_test.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="vyasmax9/tourism-prediction",
        repo_type="dataset",
    )
