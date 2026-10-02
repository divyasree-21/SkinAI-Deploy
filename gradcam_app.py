import torch
import numpy as np
import cv2

from PIL import Image
from torchvision import transforms

from student_modelgn import create_student_model


# --------------------------------------------------
# Model configuration
# --------------------------------------------------

MODEL_PATH = "Models/student_model_gn_distilled.pth"

CLASS_NAMES = ["AK", "BCC"]

device = torch.device("cpu")


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = create_student_model(len(CLASS_NAMES))

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint)

model.eval()


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# Grad-CAM
# --------------------------------------------------

def generate_gradcam(image):

    image = image.convert("RGB")

    original_image = image.copy()

    input_tensor = transform(image).unsqueeze(0)

    activations = []
    gradients = []


    # Target layer
    target_layer = model.inception5b


    def forward_hook(module, input, output):
        activations.append(output)


    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])


    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )


    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------

    output = model(input_tensor)

    predicted_index = int(torch.argmax(output, dim=1).item())


    # --------------------------------------------------
    # Backward pass
    # --------------------------------------------------

    model.zero_grad()

    score = output[0, predicted_index]

    score.backward()


    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()


    # --------------------------------------------------
    # Get activation and gradient
    # --------------------------------------------------

    activation = activations[0].detach()

    gradient = gradients[0].detach()


    # --------------------------------------------------
    # Calculate Grad-CAM weights
    # --------------------------------------------------

    weights = gradient.mean(
        dim=(2, 3),
        keepdim=True
    )


    cam = (weights * activation).sum(
        dim=1
    ).squeeze()


    # --------------------------------------------------
    # ReLU
    # --------------------------------------------------

    cam = torch.relu(cam)

    cam = cam.numpy()


    # Normalize
    cam = cam - cam.min()

    if cam.max() != 0:
        cam = cam / cam.max()


    # --------------------------------------------------
    # Resize heatmap
    # --------------------------------------------------

    cam = cv2.resize(
        cam,
        original_image.size
    )


    heatmap = np.uint8(
        255 * cam
    )


    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )


    # Convert RGB image to OpenCV format
    original_array = np.array(
        original_image
    )

    original_array = cv2.cvtColor(
        original_array,
        cv2.COLOR_RGB2BGR
    )


    # --------------------------------------------------
    # Overlay
    # --------------------------------------------------

    overlay = cv2.addWeighted(
        original_array,
        0.6,
        heatmap,
        0.4,
        0
    )


    # Convert back to RGB
    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB
    )


    return (
        predicted_index,
        heatmap,
        overlay
    )