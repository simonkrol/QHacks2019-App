#!/bin/env python3.6
# Simon Krol, Feb 2019
# Maintain the model used to classify urban sound data
# Works for .wav files

import os
import pickle
import math
import numpy as np
from file import File
from sklearn.ensemble import RandomForestClassifier


class Model:
    def __init__(self, model_location, interval_len, class_labels):
        self.loaded_model = pickle.load(open(model_location, 'rb'))
        self.interval_len = interval_len
        self.class_labels = class_labels

        # Since sound not being found is now classified as a -1,
        # Make the last element in our class labels 'sound not identified'
        self.class_labels.append("No Identifiable Sound")
        self.string_prediction = []



    def load_file(self, file_location):
        """ Load the given file to our model

        Loads the file at the given location into our model to be processed

        Arguments:
        file_location -- The name and location of the file being read
        """
        self.cur_file = File(file_location, self.interval_len)
        self.string_prediction = []

    def get_prediction(self):
        """ Get the prediction associated with the loaded file

        Get the prediction associated with the loaded file, if a new file hasn't
        been loaded since the last time this was run, return the last prediction.

        Returns the prediction as a list of Strings
        """
        if(any(self.string_prediction)):
            return self.string_prediction
        prediction = self.loaded_model.predict_proba(self.cur_file.formed_data)
        encoded_prediction = self.cur_file.format_prediction(prediction)
        self.string_prediction = self._convert_to_string(encoded_prediction)
        return self.string_prediction

    def _convert_to_string(self, prediction):
        """ Convert the predictions from indices to Strings

        Given a list of run length encoded values, convert to a list of Strings
        representing the model's predictions on the current file

        Arguments:
        prediction -- The list of run length encoded values

        Returns the prediction as a list of strings
        """
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


