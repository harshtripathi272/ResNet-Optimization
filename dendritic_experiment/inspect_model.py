import torch
import torchvision.models as models

def inspect_resnet18():
    model = models.resnet18()
    print("Model Architecture Inspection:")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(f"Conv2d: {name}")
        elif isinstance(module, torch.nn.Linear):
            print(f"Linear: {name}")

if __name__ == "__main__":
    inspect_resnet18()
