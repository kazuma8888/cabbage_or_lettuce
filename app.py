from flask import Flask, render_template, request
import cv2
import io
import numpy as np
from predict import main

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index_get():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        img = request.files['file']
        img = img.stream.read()
        bin_data = io.BytesIO(img)
        file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        result = main(img)
        print(result)
        return render_template('result.html', result = result)

if __name__=='__main__':
    app.run()