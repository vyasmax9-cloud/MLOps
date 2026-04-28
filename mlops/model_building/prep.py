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
repo_id = "vyasmax9/tourism-app"
repo_type = "dataset"
DATASET_PATH = f"hf://datasets/{repo_id}/tourism.csv"
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

# Create the local directory to store prepared data
local_prepared_data_folder = "tourism_project/data"
os.makedirs(local_prepared_data_folder, exist_ok=True)

# Save the split datasets into this local folder
X_train.to_csv(os.path.join(local_prepared_data_folder, "Xtrain.csv"), index=False)
X_test.to_csv(os.path.join(local_prepared_data_folder, "Xtest.csv"), index=False)
y_train.to_csv(os.path.join(local_prepared_data_folder, "ytrain.csv"), index=False)
y_test.to_csv(os.path.join(local_prepared_data_folder, "ytest.csv"), index=False)

# Upload the entire folder containing the prepared data
api.upload_folder(
    folder_path=local_prepared_data_folder,
    repo_id=repo_id,
    repo_type=repo_type,
)
print(f"Prepared data (Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv) uploaded to '{repo_id}'")
