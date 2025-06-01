import os, shutil, dotenv, tempfile
from huggingface_hub import HfApi, login
from pathlib import Path
from utils import *

class HuggingFaceRepoManager:
    def __init__(self):
        # Get environment variables
        dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
        self.session_name = os.getenv("session_name")
        self.huggingface_read = os.getenv('huggingface_read')
        self.huggingface_write = os.getenv('huggingface_write')
        self.repo_id = os.getenv('repo_ID')
        
        # Validate required environment variables
        if not all([self.session_name, self.huggingface_read, self.huggingface_write, self.repo_id]):
            raise ValueError("Missing required environment variables: session_name, huggingface_read, huggingface_write, repo_ID")
        
        # Initialize HuggingFace API
        self.api = HfApi()
        
        # Login with write token for operations that require authentication
        login(token=self.huggingface_write)
    
    def deleteRepoContents(self, dirName=None):
        """
        Deletes repository content.
        
        Args:
            dirName (str, optional): Name of the directory to delete. 
                If __all, deletes all repository content.
        
        Note: HuggingFace Hub doesn't support direct content deletion,
        so we'll delete all files individually.
        """
        if dirName == "__all":
            command = input("are you sure (y/n)")
            if command != "y":
                print("wrong command given. exiting.")
                exit()
        try:
            # Get list of all files in the repository
            repo_files = self.api.list_repo_files(repo_id=self.repo_id)
            
            # Filter files based on dirName if provided
            if dirName:
                # Normalize directory name (ensure it ends with /)
                if not dirName.endswith('/'):
                    dirName += '/'
                
                # Filter files that are in the specified directory
                files_to_delete = [f for f in repo_files if f.startswith(dirName)]
                
                if not files_to_delete:
                    print(f"No files found in directory: {dirName}")
                    return
                    
                print(f"Found {len(files_to_delete)} files in directory '{dirName}' to delete")
            else:
                files_to_delete = repo_files
                print(f"Deleting all {len(files_to_delete)} files in repository")
            
            # Delete each file
            for file_path in files_to_delete:
                try:
                    self.api.delete_file(
                        path_in_repo=file_path,
                        repo_id=self.repo_id,
                        commit_message=f"Delete {file_path} - {self.session_name}"
                    )
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {str(e)}")
            
            if dirName:
                print(f"Directory '{dirName}' deletion completed for {self.repo_id}")
            else:
                print(f"Repository content deletion completed for {self.repo_id}")
            
        except Exception as e:
            print(f"Error deleting repository content: {str(e)}")
            raise
    
    def uploadDirectoryContents(self, dirPath, onlyHistory = False):
        """
        Uploads all contents of a directory to the repository.
        
        Args:
            dirPath (str): Path to the directory to upload
            onlyHistory (bool): If True, reads the file and gets the lstHistory 
                and uploads only it.
        """
        dirPath = Path(dirPath)
        
        if not dirPath.exists():
            raise FileNotFoundError(f"Directory not found: {dirPath}")
        
        if not dirPath.is_dir():
            raise ValueError(f"Path is not a directory: {dirPath}")
        
        try:
            if not onlyHistory:
                # Upload the entire folder
                self.api.upload_folder(
                    folder_path=str(dirPath),
                    path_in_repo = os.path.basename(os.path.normpath(dirPath)), 
                    repo_id=self.repo_id,
                    commit_message=f"Upload directory contents - {self.session_name}",
                    ignore_patterns=[".git", "__pycache__", "*.pyc", ".DS_Store"]
                )
                print(f"Successfully uploaded contents of {dirPath} to {self.repo_id}")
            else:
                __lstDir = os.listdir(dirPath)
                __info = {
                    "platform": "huggingface",
                    "api": self.api,
                    "repoID": self.repo_id,
                    "dirName": f"./{self.session_name}-{os.path.basename(os.path.normpath(dirPath))}",
                    "private": False,
                    "replace": True
                }

                for file in __lstDir:
                    # Only upload files
                    if not ".pth" in file: continue

                    file = os.path.join(dirPath, file)
                    __data = torch.load(file, weights_only = False)
                    
                    backUpToCloud(obj = __data["train_history"], objName = f"{self.session_name}-{os.path.splitext(os.path.basename(file))[0]}", info = __info)
        except Exception as e:
            print(f"Error uploading directory contents: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Create an instance of the repo manager
    repo_manager = HuggingFaceRepoManager()
    
    # Example: Delete all repository content
    repo_manager.deleteRepoContents(input("Enter directory name. Enter '__all' to clean all."))
    
    # Example: Upload a directory
    # repo_manager.uploadDirectoryContents("Data/parallelDQN_64_64_0.0001_0.9995_1000_0.995_1_", onlyHistory = True)
    
    print("HuggingFace Repository Manager initialized successfully!")
    print(f"Session: {repo_manager.session_name}")
    print(f"Repository: {repo_manager.repo_id}")