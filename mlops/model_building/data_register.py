from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os


repo_id = "vyasmax9/tourism-prediction"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Define the path to the tourism.csv file.
# In Colab, after files.upload(), it's usually at /content/tourism.csv.
# The %run command executes from /content/, so 'tourism.csv' directly references it.
# In GitHub Actions, assuming 'tourism.csv' is committed to the repo root,
# and the script is executed from the repo root, 'tourism.csv' will also directly reference it.
tourism_csv_local_path = "tourism.csv" # This refers to the file in the current working directory

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

# Upload the tourism.csv file
api.upload_file(
    path_or_fileobj=tourism_csv_local_path,
    path_in_repo="tourism.csv", # Name of the file within the Hugging Face dataset repo
    repo_id=repo_id,
    repo_type=repo_type,
)

