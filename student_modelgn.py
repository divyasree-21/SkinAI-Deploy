import torch.nn as nn

from torchvision.models import (
    googlenet,
    GoogLeNet_Weights
)


def create_student_model(num_classes):

    print("\nCreating GoogLeNet Student Model...")
    print("-" * 50)

    # Pretrained GoogLeNet
    weights = GoogLeNet_Weights.DEFAULT

    # Important:
    # Pretrained GoogLeNet weights require aux_logits=True
    model = googlenet(
        weights=weights,
        aux_logits=True
    )

    # Disable auxiliary outputs after loading weights
    model.aux_logits = False

    # Replace final classification layer
    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        num_classes
    )

    print("GoogLeNet Student Created Successfully!")
    print("Number of Classes:", num_classes)
    print("Input Size: 224 x 224")
    print("Final Layer:", model.fc)

    return model