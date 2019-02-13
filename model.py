import os
import pickle
import numpy as np
import pandas as pd
from file import File
import math

from sklearn.ensemble import RandomForestClassifier


class Model:
    def __init__(self, model_location, interval_len, tmp_location, class_labels):
        self.loaded_model = pickle.load(open(model_location, 'rb'))
        self.interval_len = interval_len
        self.tmp_location = tmp_location
        self.class_labels = class_labels
        self.string_prediction = []



    def load_file(self, file_location):
        self.cur_file = File(file_location, self.interval_len)
        self.string_prediction = []

    def get_prediction(self):
        if(any(self.string_prediction)):
            return self.string_prediction
        prediction = self.loaded_model.predict_proba(self.cur_file.formed_data)
        encoded_prediction = self.cur_file.format_prediction(prediction)
        self.string_prediction = self.convert_to_string(encoded_prediction)
        return self.string_prediction

    def convert_to_string(self, prediction):
        string_prediction = []
        time = 0
        for ind in range(0, len(prediction), 2):
            new_time = time + prediction[ind+1] -1
            if(time == new_time):
                string_prediction.append(f"{self.class_labels[prediction[ind]]} at {time}")
            else:
                string_prediction.append(f"{self.class_labels[prediction[ind]]} from {time} until {new_time}")
            time = new_time + 1
        return string_prediction


