import torch
import torch.nn as nn
import torchvision.models as models

def verify_targeting():
    print("Verifying Layer Targeting Logic...")
    model = models.resnet18()
    
    # --- LOGIC FROM train.py ---
    class TargetedConv2d(nn.Conv2d):
        """Wrapper class to target specific layers for PAI."""
        pass

    def replace_layer(module, target_name, new_class):
        """Replaces a specific layer with a new class, preserving weights."""
        parts = target_name.split('.')
        parent = module
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        target_attr = parts[-1]
        old_layer = getattr(parent, target_attr)
        
        # Create new layer
        new_layer = new_class(
            old_layer.in_channels, old_layer.out_channels,
            old_layer.kernel_size, old_layer.stride, old_layer.padding,
            old_layer.dilation, old_layer.groups,
            old_layer.bias is not None, old_layer.padding_mode
        )
        
        # Copy weights and bias
        new_layer.weight.data = old_layer.weight.data.clone()
        if old_layer.bias is not None:
            new_layer.bias.data = old_layer.bias.data.clone()
            
        # Replace
        setattr(parent, target_attr, new_layer)
        print(f"  -> Replaced: {target_name}")

    targets = [
        "layer3.0.conv2",
        "layer3.1.conv2",
        "layer4.0.conv2"
    ]
    
    for target in targets:
        replace_layer(model, target, TargetedConv2d)
    # ---------------------------

    print("\nChecking Model Structure:")
    found_targets = 0
    for name, module in model.named_modules():
        if isinstance(module, TargetedConv2d):
            print(f"  [MATCH] {name} is TargetedConv2d")
            found_targets += 1
        elif isinstance(module, nn.Conv2d) and name in targets:
             print(f"  [FAIL] {name} is still Conv2d!")

    if found_targets == 3:
        print("\nSUCCESS: All 3 targets correctly replaced.")
    else:
        print(f"\nFAILURE: Found {found_targets}/3 targets.")

if __name__ == "__main__":
    verify_targeting()
