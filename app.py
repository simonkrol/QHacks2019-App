
import librosa
from sklearn.ensemble import RandomForestClassifier
import pickle
from flask import Flask, request, render_template
from flask import send_file
import numpy as np
import pandas as pd
import math


app = Flask(__name__)


filename = "finalized_model.sav"
tmp_wav = "tmp.wav"
loaded_model = pickle.load(open(filename, 'rb'))


Classes = ['Siren', 'Street Music', 'Drilling', 'Dog Barking', 'Children Playing', 'Gun Shot', 'Engine Idling', 'Air Conditioner', 'Jackhammer', 'Car Horn']
interval_len = 4
duration = 0

@app.route('/', methods = ['GET'])
@app.route('/api', methods=['GET'])
def upload_file():
   return render_template('upload.html')
@app.route('/', methods = ['POST'])
@app.route('/api',methods=['POST'])
def api():

    global duration
    global interval_len


    file = request.files['fileInput']
    if(not file.filename.endswith(".wav")):
        return render_template('upload.html', ["Invalid File, please upload a .wav file"])

    file.save(tmp_wav, buffer_size=16384) #Here we save the file because I couldnt figure out how to load it with librosa otherwise
    new, rate = librosa.load(tmp_wav)

    interval_len = 4
    num_samples = len(new)
    duration = math.floor(librosa.get_duration(y=new, sr=rate))
    if(duration==0):
        duration=1
    if(duration<interval_len):
        interval_len = duration

    #Determine the length of frame we need to split up the file
    frame_len = math.ceil(interval_len * num_samples/duration)
    hop_len = math.floor((frame_len/interval_len))
    #The frame function creates a subset of the data, of length frame, then hops forward the hop_length before taking another subset
    y_out = librosa.util.frame(new, frame_length=frame_len, hop_length=hop_len)

    formed_data=[]
    for i in range(len(y_out[0])):
        formed_data.append(np.mean(librosa.feature.mfcc(y=y_out[:, i], sr=rate, n_mfcc=200).T, axis = 0))

    predictions = loaded_model.predict_proba(formed_data)
    predicted_string=""
    for i in range(len(predictions)):
        predicted_string+=getPrediction(predictions[i])
        predicted_string+=f",{i}:{i+interval_len} \n"

    scored_string = parse(predicted_string)
    split_score  =scored_string.split("\n")
    temp=""
    final = ""
    if(duration==1):
        final = f"{split_score[0].split(':')[1]}->{split_score[0].split(':')[0]}"
    else:
        for i in range(len(split_score)-1):
            scored_string = split_score[i].split(":")
            if(i==len(split_score)-2):
                final+=f" -> {int(split_score[i-1].split(':')[0])+1}\n"
                break
            if(scored_string[1]!=temp):
                if(temp!=""):
                    final+=f" -> {split_score[i-1].split(':')[0]}\n"
                final+=f"{scored_string[1]}: {scored_string[0]}"
                temp = scored_string[1]
    print(final.split("\n")[:-1])
    split_final = final.split("\n")
    if("" in split_final):
        split_final.remove("")
    return render_template("./upload.html", result=split_final)



def parse(st):
    scoring=[]
    for i in range(duration):
        scoring.append([])

    for a in st.split("\n"):
        if(a==""):
            continue
        split = a.split(",")
        title = split[0]
        values = split[1].split(":")
        for i in range(int(values[0]), int(values[1])):
            scoring[i].append(title)
    for i in range(duration):
        scoring[i] = score(scoring[i], i)
    st = ""
    for i in range(len(scoring)):
        st+=f"{i}:{scoring[i]}\n"
    return st

def score(vals, iteration):
    if(vals.count("No Match")>=3 or vals.count("No Match")==len(vals)):
        return "No Match"
    if(iteration>=duration-interval_len):
        return vals[len(vals)-1]
    curWinner = vals[0]
    i=1
    while(curWinner == "No Match"):
        curWinner = vals[i]
        i+=1

    return curWinner




def getPrediction(predictions):
    """ Determine if a prediction is justified, if low, assume that no sound stood out"""
    maximum = max(predictions)
    if(maximum>=0.42):
        return Classes[np.argmax(predictions)]
    else:
        return "No Match"








if __name__ == '__main__':
    app.run()
