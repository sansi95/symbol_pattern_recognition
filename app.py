from flask import Flask, render_template, request
import os
import requests
from recognize import binary_function
from PIL import Image
from image_slicer import slice


def convert(image_f):
    image_file = Image.open(image_f) # open colour image
    image_file = image_file.convert('1') # convert image to black and white
    image_file.save('image.jpg')

def is_grey_scale(img_path):
    img = Image.open(img_path).convert('RGB')
    w,h = img.size
    for i in range(w):
        for j in range(h):
            r,g,b = img.getpixel((i,j))
            if r != g != b: return False
    return True

app = Flask(__name__)

@app.route('/')
def upload():
    return render_template("options.html")

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        f.save('image.jpg')

    if is_grey_scale('image.jpg') != True:
        convert('image.jpg')

    output = binary_function('symbol')
    os.remove("image.jpg")
    return(output)

@app.route('/pattern', methods = ['POST'])
def pattern():
    if request.method == 'POST':
        f = request.files['file']
        f.save('image.jpg')

    if is_grey_scale('image.jpg') != True:
        convert('image.jpg')
    image = 'image.jpg'
    slice(image, 47)
    os.remove("image.jpg")
    output = binary_function('pattern')

    files = os.listdir('.')
    for i in files:
        if i.endswith('jpg') or i.endswith('png'):
            os.remove(i)

    return(output)

if __name__ == '__main__':
    app.run(debug = True, threaded=False)
