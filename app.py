
import PIL
from joblib  import load
import io
from PIL import Image, ImageOps
from urllib.request import urlopen
from flask import Flask , request, jsonify,render_template
import numpy as np
import scipy


def image_preprocessing(result):
        # strip the result
        with urlopen(result) as response:
                result = response.read()

        # convert it numpy array
        img = Image.open(io.BytesIO(result))
        arr = np.asarray(img)

        # make pillow image object
        arr = Image.fromarray(arr)
        # resize images to 28,28
        arr = arr.resize((28, 28))

        # make white background png
        arr = arr.convert("RGBA")
        arr1 = Image.new("RGBA", arr.size, "WHITE")
        arr1.paste(arr, mask=arr)
        arr = arr1.convert("RGB")

        # make greyscale image
        arr = arr.convert("L")
        # invert the image
        arr = PIL.ImageOps.invert(arr)
        # make numpy image array
        n_arr = np.array(arr)
        n_arr = n_arr.reshape(784)
        #inteoduce comma seperated as mnist data set
        n_arr=list(n_arr)

        return n_arr

app = Flask(__name__)
trained_model=load("ml_model/best_KNN_clf")
@app.route("/")
def index():

        return render_template('index.html')

@app.route("/predict",methods=['post'])
def predict():

        #get uri from javascript canvas as result
        result=request.form.get('text')
        #preprocessing
        n_arr=image_preprocessing(result)
        predicted_number = trained_model.predict([n_arr])

        return render_template('index.html',a="PREDICTED NUMBER = {}".format(predicted_number[0]))

@app.route("/about")
def about():
        return render_template('about.html')

if __name__ == '__main__':
   app.run()