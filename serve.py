import time
import timeit

import PIL
import pretrainedmodels
import torchvision.io
from PIL.Image import Image
from flask import Flask, request, jsonify
import torch
from torchvision import transforms
import io
import model_architecture


# Define a function for image pre-processing
def preprocess_image(pil_image: Image):
    # Load image from bytes

    # Convert to tensor
    transform = transforms.Compose([
        transforms.Resize(96),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    image_tensor = transform(pil_image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


model_dict = torch.load('./resnet34.fold0.best.pt', map_location=torch.device('cpu'))
use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")

base_model = pretrainedmodels.resnet34(num_classes=1000, pretrained='imagenet').to(device)  # load pretrained as base
model = model_architecture.Net(base_model, 512).to(device)  # create model
model.load_state_dict(model_dict)  # loading weights
model.eval()
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # Get image data from request
    if request.method == 'POST':
        image_file = request.files['image']
        image_bytes = image_file.read()

        image = PIL.Image.open(io.BytesIO(image_bytes))

        # Preprocess image
        image_tensor = preprocess_image(image)

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            print(outputs)
            _, predicted = torch.max(outputs.data, 1)

        # Return prediction as JSON
        values = {
            0: "No Cancer Detected",
            1: "Cancer Detected",
        }
        return jsonify({'prediction': values[predicted.item()], "probability": outputs.data.tolist()[0][0]})


if __name__ == '__main__':
    app.run(debug=True)
