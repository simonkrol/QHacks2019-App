#!/bin/env python3.6
# Simon Krol, Feb 2019
# Maintain the files being classified by our models
import numpy as np
import librosa
from itertools import chain, groupby
import math

class File:
    def __init__(self, file_location, interval_len):
        self.new, self.rate = librosa.load(file_location)
        self.duration = self._get_duration()
        self.interval_len = min(self.duration, interval_len)

        n_samples = len(self.new)
        self.y_out = self._split_file(n_samples)
        self.formed_data = self._form_data()
        self.threshold = 0.42


    def format_prediction(self, predictions):
        """ Format our prediction

        Given our percentage predictions for each chunk of the file, format the predictions as
        run length encoded values.

        Arguments:
        predictions The percentage based predictions for each interval_len chunk of our file.

        Returns the predictions as a list of run length encoded values.
        """
        index_split_prediction = [self._classify_prediction(prediction) for prediction in predictions]
        index_merged_prediction = self._merge_predictions(index_split_prediction)
        scored_prediction = [self._score_prediction(pred, it, -1) for it, pred in enumerate(index_merged_prediction)]
        encoded_prediction = self._run_length_encode(scored_prediction)
        return encoded_prediction

    def _get_duration(self):
        """ Get the duration of the file

        Get the duration of the file, if duration is 0, add 1.

        Returns the duration of the file as an integer
        """
        duration = math.floor(librosa.get_duration(y=self.new, sr=self.rate))
        duration += (duration == 0)
        return duration

    def _split_file(self, n_samples):
        """ Split the file into interval_len size chunks

        Split the file in interval_len second long chunks to be processed individually in
        order to classify different parts of the file.

        Arguments:
        n_samples -- The number of samples throughout the entire file

        Returns a list of the values associated with each chunk
        """

        #Determine the length of frame we need to split up the file
        #This is equal to the number of frames found within 1 interval length(in s)
        frame_len = math.ceil(self.interval_len * n_samples/self.duration)

        #This is equal to the number of frames found within 1 second
        hop_len = math.floor((frame_len/self.interval_len))
        #The frame function creates a subset of the data, of length frame, then hops forward the hop_length before taking another subset
        return librosa.util.frame(self.new, frame_length=frame_len, hop_length=hop_len)

    def _form_data(self):
        """ Convert the data to the average of its mfcc form

        Convert the data to its Mel-Frequency Cepstral Coefficients (MFCC) form

        Returns the converted data
        """
        formed_data=[]
        for i in range(len(self.y_out[0])):
            formed_data.append(np.mean(librosa.feature.mfcc(y=self.y_out[:, i], sr=self.rate, n_mfcc=200).T, axis = 0))
        return formed_data

    def _classify_prediction(self, prediction):
        """ Classify each prediction based on the max prediction

        Given our percentage predictions for each chunk of the file,
        determine if the highest prediction surpasses our threshold.


        Arguments:
        predictions The percentage based predictions for each interval_len chunk of our file.

        Returns the predictions as a list of indices representing the chosen prediction for each chunk
        """
        max_index = np.argmax(prediction)
        if(prediction[max_index] >= self.threshold):
            return max_index
        return -1

    def _merge_predictions(self, prediction):
        """ Determine all the predictions for each second of the file

        Due to our interval length, a prediction is made about a (usually 4)
        second value and each second can have up to 4 predictions made about it.
        This function determines all the predictions made about a value.
        For example: If the prediction for 0->3 is 7, and the prediction for 1->4
        is 4, 0 will contain the prediction 7, 1,2 and 3 will contain the prediction
        7 and 4, and 4 will contain only the prediction 4.

        Arguments:
        predictions -- The maxed predictions for each interval_len chunk of our file.

        Returns the predictions as a 2 dimensional list of all predictions for each second
        """
        merged_prediction = []
        for second in range(self.duration):
            merged_prediction.append(prediction[max(0, second + 1 - self.interval_len):second+1])
        return merged_prediction

    def _score_prediction(self, prediction, iteration, negative):
        """ Based on all the predictions for a second, choose a winner

        The rules for choosing a winner are based around a sort of first come first serve
        mentality, where the first value predicted is weighed more heavily than the others,
        The only way that a first predicted value can be beaten is if were looking at the last
        few predictions(That way we dont lose out on our final predictions) or if the negative
        case (No sound classified) maintains the entire rest of the prediction list.

        Arguments:
        prediction -- The indices of the predictions for a given second
        iteration -- Which second we're looking at, used to determine if we're at
        the last few seconds.
        negative -- The value we should consider to be the null or negative case.

        Returns an integer representing the winning prediction
        """
        neg_count = prediction.count(negative)
        if((neg_count>=self.interval_len-1 and neg_count != 0 ) or neg_count==len(prediction)):
            return negative
        if(iteration>=self.duration-self.interval_len):
            return prediction[-1]
        return [pred for pred in prediction if pred != negative][0]

    def _run_length_encode(self, prediction):
        """ Encode the predictions in run length form

        The predictions are currently in the form where we have a winner for each
        second, this function converts that to run-length form, where we identify
        the winning predictions, followed by the number of consecutive seconds they
        were winning. This is done for the entire list of predictions.
        Ex: [1,1,1,2,2,3,5,5,1,1] becomes [1,3,2,2,3,1,5,2,1,2] indicating 3 ones, 2 twos,
        1 three, 2 fives and 2 ones.

        Arguments:
        prediction -- The winning prediction value for each second of our sound file

        Returns The run length encoded values of our prediction in a list
        """
        return list(chain.from_iterable(
            (val, len([*thing]))
            for val, thing in groupby(prediction)
        ))
