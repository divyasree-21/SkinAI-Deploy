import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms


MODEL_PATH = "Models/student_model_gn_distilled.onnx"

# Keep these in the exact order used when training the model.
# These labels are used for prediction indexing and display.
CLASS_NAMES = [
    "AK (Actinic Keratosis)",
    "BCC (Basal Cell Carcinoma)",
]

# Mapping is useful when older history records or other code still use
# the short labels "AK" and "BCC".
DISPLAY_NAMES = {
    "AK": "AK (Actinic Keratosis)",
    "BCC": "BCC (Basal Cell Carcinoma)",
    "AK (Actinic Keratosis)": "AK (Actinic Keratosis)",
    "BCC (Basal Cell Carcinoma)": "BCC (Basal Cell Carcinoma)",
}


# Load the ONNX model for local CPU inference.
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"],
)

INPUT_NAME = session.get_inputs()[0].name
OUTPUT_NAME = session.get_outputs()[0].name


# Image preprocessing must match the preprocessing used for the model.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


def predict(image: Image.Image) -> np.ndarray:
    """Return the model output scores for a PIL image.

    The caller can apply softmax to these scores to calculate probabilities.
    The returned score order corresponds to CLASS_NAMES.
    """
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    input_data = image_tensor.numpy().astype(np.float32)

    output = session.run(
        [OUTPUT_NAME],
        {INPUT_NAME: input_data},
    )[0]

    return np.asarray(output[0])
