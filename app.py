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

    return model.predict(image, conf=0.25, device='cpu')


def _count(image_string):
    results = predict(image_string)
    result = results[0]

    classes = result.boxes.cls
    names = result.names

    unique, counts = np.unique(classes.cpu(), return_counts=True)
    unique = map(lambda x: names[x], unique)
    counts = map(lambda x: int(x), counts)

    return dict(zip(unique, counts))


def _infer(image_string):
    results = predict(image_string)
    result = results[0]
    classes = result.names

    out = {
        'classes': classes,
        'objects': []
    }

    objects = result.boxes.data.cpu().tolist()

    for object in objects:
        out['objects'].append({
            'x0': object[0],
            'y0': object[1],
            'x1': object[2],
            'y1': object[3],
            'confidence': object[4],
            'class': object[5]
        })

    return out


@app.post('/count')
def count():
    image = request.json['image']
    return _count(image)


@app.post('/predict')
def infer():
    image = request.json['image']
    return _infer(image)


if __name__ == "__main__":
    print('hello, world!')
    app.run(debug=True, host="0.0.0.0", port=5000)
