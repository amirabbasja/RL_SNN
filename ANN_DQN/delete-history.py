import torch
import os
import collections

def load_and_print_pytorch_dict(file_path):
    """
    Load a PyTorch saved dictionary and print its contents.
    
    Args:
        file_path (str): Path to the PyTorch saved file (.pt or .pth)
    """
    # Check if file exists
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return
    
    try:
        # Load the dictionary
        loaded_obj = torch.load(file_path, weights_only = False) 
        
        for item in loaded_obj[-1].keys():
            print(item)
        # print(f"Successfully loaded object from {file_path}")
        
        # # Handle different types of loaded objects
        # if isinstance(loaded_obj, dict):
        #     print("\nLoaded object is a dictionary")
        #     print("\nKeys in the dictionary:")
        #     for key in loaded_obj.keys():
        #         print(f"- {key}")
            
        #     print("\nDetailed contents:")
        #     for key, value in loaded_obj.items():
        #         if isinstance(value, torch.Tensor):
        #             print(f"{key}: Tensor of shape {value.shape}, dtype {value.dtype}")
        #         else:
        #             print(f"{key}: {type(value)} - {value}")
        
        # elif isinstance(loaded_obj, collections.deque):
        #     print("\nLoaded object is a collections.deque")
        #     print(f"Length: {len(loaded_obj)}")
        #     print("\nDetailed contents:")
        #     for i, item in enumerate(loaded_obj):
        #         if isinstance(item, torch.Tensor):
        #             print(f"Item {i}: Tensor of shape {item.shape}, dtype {item.dtype}")
        #         else:
        #             print(f"Item {i}: {type(item)} - {item}")
        
        # else:
        #     print(f"\nLoaded object is of type: {type(loaded_obj)}")
        #     if hasattr(loaded_obj, "__len__"):
        #         print(f"Length: {len(loaded_obj)}")
        #     print("\nObject content:")
        #     print(loaded_obj)
                
    except Exception as e:
        print(f"Error loading the file: {e}")

if __name__ == "__main__":
    # Replace with your actual file path
    file_path = "./Data/parallelDQN_64_64_0.0001_0.999_1000_0.995__backup.pth"
    load_and_print_pytorch_dict(file_path)