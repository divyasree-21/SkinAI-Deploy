import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms


MODEL_PATH = "Models/student_model_gn_distilled.onnx"

CLASS_NAMES = ["AK", "BCC"]


# Load ONNX model
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

INPUT_NAME = session.get_inputs()[0].name
OUTPUT_NAME = session.get_outputs()[0].name


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


def predict(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Apply preprocessing
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Convert PyTorch tensor to NumPy
    input_data = image_tensor.numpy().astype(np.float32)

    # Run ONNX inference
    output = session.run(
        [OUTPUT_NAME],
        {INPUT_NAME: input_data}
    )[0]

    return output[0]