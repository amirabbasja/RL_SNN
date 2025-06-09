import dotenv, os
from data_manager import *

"""
The data for accounts are acquired from the .env file. Each account should have 
session_name, huggingface_read, huggingface_write, repo_ID in one string, separated by a comma. 
Variable names should be acc0, acc1, acc2, etc. Following is an example:

acc0 = Amir_Abbas,<API_KEY>,<API_KEY>,abbasJa123/ANN_DQN
acc1 = cwolf480,<API_KEY>,<API_KEY>,cwolf480/ANN_DQN
acc2 = ryamwokkins,<API_KEY>,<API_KEY>,ryanwokking/ANN_DQN
"""

# Read all dot env variables
# Load the .env file
dotenv.load_dotenv()

# Iterate through all environment variables
for key, value in os.environ.items():
    if "acc" in key:
        accountNum = key.replace("acc","")
        sessionName, huggignface_read, huggignface_write, repoId = value.split(",")
        manager = HuggingFaceRepoManager(sessionName, huggignface_read, huggignface_write, repoId)
        manager.downloadRepoContents(file_pattern = "*pth")