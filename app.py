from ultralytics import YOLO
import numpy as np
import base64
from PIL import Image
from io import BytesIO
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = YOLO('weights/best.pt')

def predict(image_string):
    image_bytes = base64.b64decode(image_string)
    image = Image.open(BytesIO(image_bytes))

    results = model.predict(image, conf=0.25, device='cpu')

    classes = results[0].boxes.cls
    names = results[0].names

    unique, counts = np.unique(classes.cpu(), return_counts=True)
    unique = map(lambda x: names[x], unique)
    counts = map(lambda x: int(x), counts)

    out = dict(zip(unique, counts))
    return out


@app.post('/count')
def count():
    image = request.json['image']
    return predict(image)


if __name__ == "__main__":
    print('hello, world!')
    app.run(debug=True, host="0.0.0.0", port=5000)
