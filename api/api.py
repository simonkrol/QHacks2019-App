import librosa
from sklearn.ensemble import RandomForestClassifier
import pickle
from flask import Flask, request
from flask import send_file
import numpy as np
import pandas as pd
import math


app = Flask(__name__)
filename = "finalized_model.sav"
s = "tmp.wav"
loaded_model = pickle.load(open(filename, 'rb'))

df = pd.read_csv('../../input/train.csv')
Classes = df.Class.unique().tolist()
intervalLen = 4
duration = 0

@app.route('/',methods=['GET'])
def hello_world():
    return 'Hello World!'

@app.route('/customerupdate',methods=['GET','POST'])
def customerupdate():
    global duration
    global intervalLen
    file = request.files['document']
    file.save(s, buffer_size=16384)

    new, rate = librosa.load(s)

    intervalLen = 4
    num_samples = len(new)
    duration = math.floor(librosa.get_duration(y=new, sr=rate))
    if(duration<intervalLen):
        intervalLen = duration

    frame_len = math.ceil(intervalLen * num_samples/duration)

    hop_len = math.floor((frame_len/intervalLen))
    y_out = librosa.util.frame(new, frame_length=frame_len, hop_length=hop_len)
    mf=[]
    for i in range(len(y_out[0])):

        mf.append(np.mean(librosa.feature.mfcc(y=y_out[:, i], sr=rate, n_mfcc=200).T, axis = 0))
    a = ""
    for i in range(len(mf)):
        a+=(getPrediction(loaded_model.predict_proba([mf[i]])[0]))
        a+=f",{i}:{i+intervalLen} \n"


    return parse(a)

def parse(str):
    scoring=[]
    for i in range(duration):
        scoring.append([])

    for a in str.split("\n"):
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
    if(vals.count("IDLE")>=3):
        return "IDLE"
    if(iteration>=duration-intervalLen):
        return vals[len(vals)-1]
    curWinner = vals[0]
    i=1
    while(curWinner == "IDLE"):
        curWinner = vals[i]
        i+=1

    return curWinner




def getPrediction(predictions):
    maximum = max(predictions)
    if(maximum>=0.4):
        return Classes[np.argmax(predictions)]
    else:
        return "IDLE"
if __name__ == '__main__':
   app.run(debug = True)
