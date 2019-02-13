#!/bin/env python3.6
# Simon Krol, Feb 2019
# Classify urban sounds from sound data
# Works for .wav files

from flask import Flask, request, render_template, send_file
from model import Model

app = Flask(__name__)


# Customizable variables

# The name and location of the trained classifier, expected to be a RandomTreeClassifier
model_filename = "finalized_model.sav"
# The names of the classes being classified
classes = ['Siren', 'Street Music', 'Drilling', 'Dog Barking', 'Children Playing', 'Gun Shot', 'Engine Idling', 'Air Conditioner', 'Jackhammer', 'Car Horn']

# The name and location of the temporary file used to load the file into librosa
tmp_filename = "tmp.wav"

# The name and location of the html file rendered when the application is accessed from its webpage
render_filename = "upload.html"

# The length of the intervals we use to classify the data, interval length of 4
# indicates we split up the sound data into 4 second chunks. For example, a 6 second
# clip would be split into [1,2,3,4], [2,3,4,5] and [3,4,5,6]
interval_len = 4



model = Model(model_filename, interval_len, classes)


@app.route('/', methods = ['GET'])
@app.route('/api', methods=['GET'])
def upload_file():
    """ Render the main upload file webpage

    Respond to a get request on either the base url or the /api endpoint with the
    render of our main file upload webpage.

    Returns the render of the file upload webpage
    """
    return render_template(render_filename)


@app.route('/', methods = ['POST'])
@app.route('/api',methods=['POST'])
def api():
    """ Render the results of a given file upload

    Respond to a post request on either the base url or the /api endpoint with the
    results of a file upload

    Returns the render of the file upload webpage along with the results of the file upload
    """
    file = request.files['fileInput']

    if(not file.filename.endswith(".wav")):
        return render_template(render_filename, ["Invalid File, please upload a .wav file"])

    file.save(tmp_filename, buffer_size=16384)
    model.load_file(tmp_filename)
    return render_template(render_filename, result=model.get_prediction())


if __name__ == '__main__':
    app.run(debug = True)
