from flask import Flask, render_template, request, redirect, url_for
import torch
from PIL import Image
from torchvision import transforms
import io

app = Flask(__name__)

# Load your model (ensure this is consistent with your model's architecture)
model = load_model('path_to_downloaded_model.pth', num_classes)
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        predictions = torch.sigmoid(outputs.logits).cpu().numpy() > 0.5

    # Convert predictions to a readable format
    response = 'Your prediction results: ' + ', '.join([str(i) for i, x in enumerate(predictions[0]) if x])
    return response

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            image_bytes = file.read()
            prediction = predict(image_bytes)
            return render_template('result.html', prediction=prediction)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
