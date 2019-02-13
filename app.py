from flask import Flask, request, render_template, send_file
from model import Model

app = Flask(__name__)

filename = "finalized_model.sav"
tmp_wav = "tmp.wav"
interval_len = 4
classes = ['Siren', 'Street Music', 'Drilling', 'Dog Barking', 'Children Playing', 'Gun Shot', 'Engine Idling', 'Air Conditioner', 'Jackhammer', 'Car Horn']

model = Model(filename, interval_len, classes)


@app.route('/', methods = ['GET'])
@app.route('/api', methods=['GET'])
def upload_file():
   return render_template('upload.html')


@app.route('/', methods = ['POST'])
@app.route('/api',methods=['POST'])
def api():
    file = request.files['fileInput']

    if(not file.filename.endswith(".wav")):
        return render_template('upload.html', ["Invalid File, please upload a .wav file"])

    file.save(tmp_wav, buffer_size=16384)
    model.load_file(tmp_wav)
    return render_template("./upload.html", result=model.get_prediction())


if __name__ == '__main__':
    app.run(debug = True)
