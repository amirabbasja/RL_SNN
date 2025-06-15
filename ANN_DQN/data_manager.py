import os, shutil, dotenv, tempfile
from huggingface_hub import HfApi, login
from pathlib import Path
from utils import *
import dotenv

class HuggingFaceRepoManager:
    def __init__(self, sessionName, huggingfaceRead, huggingfaceWrite, repoID):
        """
        Args: 
            sessionName (str): Name of the session
            huggingfaceRead (str): HuggingFace read token
            huggingfaceWrite (str): HuggingFace write token
            repoID (str): HuggingFace repository ID
        """
        # Set necessary variables
        self.session_name = sessionName
        self.huggingface_read = huggingfaceRead
        self.huggingface_write = huggingfaceWrite
        self.repo_id = repoID
        
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
                dirName = dirName.strip()
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
                files_to_delete = [f for f in repo_files if f.startswith(dirName)] if dirName != "__all/" else repo_files if dirName == "__all/" else []

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
    
    def uploadDirectoryContents(self, pathsToUpload, onlyHistory = False):
        """
        Uploads all contents of a directory to the repository.
        
        Args:
            pathsToUpload (str or list): Path to the directory to upload. Can 
                a list of paths as well.
            onlyHistory (bool): If True, reads the file and gets the lstHistory 
                and uploads only it.
        """
        if type(pathsToUpload) != list:
            pathsToUpload = [pathsToUpload]
        
        for _path in pathsToUpload:
            print(f"Processing {_path}...")
            dirPath = Path(_path)
            
            if not dirPath.exists():
                raise FileNotFoundError(f"Directory not found: {dirPath}")
            
            if not dirPath.is_dir():
                raise ValueError(f"Path is not a directory: {dirPath}")
            
            try:
                if not onlyHistory:
                    # Upload the entire folder
                    try:
                        self.api.upload_folder(
                            folder_path=str(dirPath),
                            path_in_repo = os.path.basename(os.path.normpath(dirPath)), 
                            repo_id=self.repo_id,
                            commit_message=f"Upload directory contents - {self.session_name}",
                            ignore_patterns=[".git", "__pycache__", "*.pyc", ".DS_Store"]
                        )
                        print(f"Successfully uploaded contents of {dirPath} to {self.repo_id}")
                    except Exception as e:
                        print(f"Error uploading directory contents: {str(e)}")
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
                        __data = {"train_history": __data["train_history"], "elapsedTime": __data["elapsedTime"]}
                        
                        backUpToCloud(obj = __data, objName = f"{self.session_name}-{os.path.splitext(os.path.basename(file))[0]}", info = __info)
            except Exception as e:
                print(f"Error uploading directory contents: {str(e)}")
                raise
    
    def downloadRepoContents(self, local_dir=None, repo_path=None, file_pattern=None, force_download=False):
        """
        Downloads repository content to local machine.
        
        Args:
            local_dir (str, optional): Local directory to download files to. 
                Defaults to './downloads/{repo_id}'
            repo_path (str, optional): Specific path/directory in the repo to download.
                If None, downloads entire repository.
            file_pattern (str, optional): Pattern to filter files (e.g., '*.pth', '*.json').
                If None, downloads all files.
            force_download (bool, optional): If True, download files even if they already exist locally.
                If False, skip files that already exist. Defaults to False.
        
        Returns:
            str: Path to the downloaded content
        """
        try:
            # Set default local directory
            if local_dir is None:
                local_dir = f"./downloads/{self.repo_id.replace('/', '_')}"
            
            # Create local directory if it doesn't exist
            local_path = Path(local_dir)
            local_path.mkdir(parents=True, exist_ok=True)
            
            # Get list of all files in the repository
            repo_files = self.api.list_repo_files(repo_id=self.repo_id)
            
            # Filter files based on repo_path if provided
            if repo_path:
                # Normalize path (ensure it ends with / for directories)
                if not repo_path.endswith('/') and not '.' in os.path.basename(repo_path):
                    repo_path += '/'
                repo_files = [f for f in repo_files if f.startswith(repo_path)]
                
                if not repo_files:
                    print(f"No files found in repository path: {repo_path}")
                    return str(local_path)
            
            # Filter files based on pattern if provided
            if file_pattern:
                import fnmatch
                repo_files = [f for f in repo_files if fnmatch.fnmatch(f, file_pattern)]
                
                if not repo_files:
                    print(f"No files found matching pattern: {file_pattern}")
                    return str(local_path)
            
            print(f"Found {len(repo_files)} files to download")
            
            # Download each file
            downloaded_count = 0
            skipped_count = 0
            for file_path in repo_files:
                try:
                    # Create local file path
                    local_file_path = local_path / file_path
                    
                    # Check if file already exists
                    if not force_download and local_file_path.exists():
                        print(f"Skipped (already exists): {file_path}")
                        skipped_count += 1
                        continue
                    
                    # Create parent directories if they don't exist
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Download the file
                    self.api.hf_hub_download(
                        repo_id=self.repo_id,
                        filename=file_path,
                        local_dir=str(local_path),
                        local_dir_use_symlinks=False,
                        force_download=force_download
                    )
                    
                    print(f"Downloaded: {file_path}")
                    downloaded_count += 1
                    
                except Exception as e:
                    print(f"Failed to download {file_path}: {str(e)}")
            
            print(f"Download completed: {downloaded_count} files downloaded, {skipped_count} files skipped to {local_path}")
            return str(local_path)
            
        except Exception as e:
            print(f"Error downloading repository content: {str(e)}")
            raise
    
    def downloadSpecificFiles(self, file_list, local_dir=None, force_download=False):
        """
        Downloads specific files from the repository.
        
        Args:
            file_list (list): List of file paths in the repository to download
            local_dir (str, optional): Local directory to download files to.
                Defaults to './downloads/{repo_id}'
            force_download (bool, optional): If True, download files even if they already exist locally.
                If False, skip files that already exist. Defaults to False.
        
        Returns:
            str: Path to the downloaded content
        """
        try:
            # Set default local directory
            if local_dir is None:
                local_dir = f"./downloads/{self.repo_id.replace('/', '_')}"
            
            # Create local directory if it doesn't exist
            local_path = Path(local_dir)
            local_path.mkdir(parents=True, exist_ok=True)
            
            print(f"Downloading {len(file_list)} specific files")
            
            # Download each file
            downloaded_count = 0
            skipped_count = 0
            for file_path in file_list:
                try:
                    # Create local file path
                    local_file_path = local_path / file_path
                    
                    # Check if file already exists
                    if not force_download and local_file_path.exists():
                        print(f"Skipped (already exists): {file_path}")
                        skipped_count += 1
                        continue
                    
                    # Create parent directories if they don't exist
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Download the file
                    self.api.hf_hub_download(
                        repo_id=self.repo_id,
                        filename=file_path,
                        local_dir=str(local_path),
                        local_dir_use_symlinks=False,
                        force_download=force_download
                    )
                    
                    print(f"Downloaded: {file_path}")
                    downloaded_count += 1
                    
                except Exception as e:
                    print(f"Failed to download {file_path}: {str(e)}")
            
            print(f"Download completed: {downloaded_count} files downloaded, {skipped_count} files skipped to {local_path}")
            return str(local_path)
            
        except Exception as e:
            print(f"Error downloading specific files: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Initialize the repo manager
    dotenv.load_dotenv()
    env = os.environ

    sessionName = env.get("session_name", None)
    huggingfaceRead = env.get("huggingface_read", None)
    huggingfaceWrite = env.get("huggingface_write", None)
    repoID = env.get("repo_ID", None)
    
    # Create an instance of the repo manager
    repo_manager = HuggingFaceRepoManager(sessionName, huggingfaceRead, huggingfaceWrite, repoID)
    
    # Example: Delete all repository content
    # repo_manager.deleteRepoContents(input("Enter directory name. Enter '__all' to clean all."))
    
    # Example: Upload a directory
    repo_manager.uploadDirectoryContents([os.path.join("./Data", _p) for _p in os.listdir("./Data")], onlyHistory = True)
    
    # Example: Download repository content
    
    print("HuggingFace Repository Manager initialized successfully!")
    print(f"Session: {repo_manager.session_name}")
    print(f"Repository: {repo_manager.repo_id}")