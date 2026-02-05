import sys
import os
import torch

# Add current dir to path
sys.path.append(os.path.join(os.getcwd(), 'dendritic_experiment'))

import config
import models
import utils

def test_setup():
    print("Testing setup...")
    
    # 1. Check PAI Credentials
    print(f"PAI_EMAIL: {config.PAI_EMAIL}")
    print(f"PAI_TOKEN: {config.PAI_TOKEN}")
    if "INSERT" in config.PAI_EMAIL or "INSERT" in config.PAI_TOKEN:
        print("WARNING: PAI credentials appear to be placeholders. Please update config.py.")
    
    # 2. Check PAI Import
    sys.path.append(config.PAI_REPO_PATH)
    try:
        import perforatedai
        print("SUCCESS: PerforatedAI imported.")
    except ImportError:
        print(f"ERROR: Could not import perforatedai from {config.PAI_REPO_PATH}")
    
    # 3. Check Data Loading (Dry Run)
    try:
        trainloader, testloader = utils.get_dataloaders(batch_size=4, num_workers=0)
        print("SUCCESS: Dataloaders created.")
        
        # Fetch one batch
        images, labels = next(iter(trainloader))
        print(f"Batch shape: {images.shape}")
        if images.shape[2:] != (224, 224):
            print(f"ERROR: Image size is {images.shape[2:]}, expected (224, 224)")
        else:
            print("SUCCESS: Image resizing correct.")
            
    except Exception as e:
        print(f"ERROR: Data loading failed: {e}")
        
    # 4. Check Models
    try:
        model = models.get_resnet18()
        print("SUCCESS: ResNet18 initialized.")
        model = models.get_resnet34()
        print("SUCCESS: ResNet34 initialized.")
    except Exception as e:
        print(f"ERROR: Model initialization failed: {e}")

    print("\nSetup test complete.")

if __name__ == "__main__":
    test_setup()
