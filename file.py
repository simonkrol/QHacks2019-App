import numpy as np
import librosa
from itertools import chain, groupby
# import pandas as pd
import math

class File:
    def __init__(self, file_location, interval_len):
        self.new, self.rate = librosa.load(file_location)
        self._get_duration()
        self.interval_len = min(self.duration, interval_len)

        n_samples = len(self.new)
        self.y_out = self._split_file(n_samples)
        self.formed_data = self._form_data()
        self.threshold = 0.42

    def _get_duration(self):
        self.duration = math.floor(librosa.get_duration(y=self.new, sr=self.rate))
        self.duration += (self.duration == 0)

    def _split_file(self, n_samples):
        #Determine the length of frame we need to split up the file
        #This is equal to the number of frames found within 1 interval length(in s)
        frame_len = math.ceil(self.interval_len * n_samples/self.duration)

        #This is equal to the number of frames found within 1 second
        hop_len = math.floor((frame_len/self.interval_len))
        #The frame function creates a subset of the data, of length frame, then hops forward the hop_length before taking another subset
        return librosa.util.frame(self.new, frame_length=frame_len, hop_length=hop_len)

    def _form_data(self):
        formed_data=[]
        for i in range(len(self.y_out[0])):
            formed_data.append(np.mean(librosa.feature.mfcc(y=self.y_out[:, i], sr=self.rate, n_mfcc=200).T, axis = 0))
        return formed_data

    def format_prediction(self, predictions):
        index_split_prediction = [self._classify_prediction(prediction) for prediction in predictions]
        index_merged_prediction = self._merge_predictions(index_split_prediction)
        scored_prediction = [self._score_prediction(pred, it, -1) for it, pred in enumerate(index_merged_prediction)]
        encoded_prediction = self._run_length_encode(scored_prediction)
        return encoded_prediction

    def _classify_prediction(self, prediction):
        max_index = np.argmax(prediction)
        if(prediction[max_index] >= self.threshold):
            return max_index
        return -1

    def _merge_predictions(self, prediction):

        merged_prediction = []
        for second in range(self.duration):
            merged_prediction.append(prediction[max(0, second + 1 - self.interval_len):second+1])
        return merged_prediction

    def _score_prediction(self, prediction, iteration, negative):
        neg_count = prediction.count(negative)
        if(neg_count>=self.interval_len-1 or neg_count==len(prediction)):
            return negative
        if(iteration>=self.duration-self.interval_len):
            return prediction[-1]
        return [pred for pred in prediction if pred != negative][0]

    def _run_length_encode(self, prediction):
        return list(chain.from_iterable(
            (val, len([*thing]))
            for val, thing in groupby(prediction)
        ))
