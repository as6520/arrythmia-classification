from ecg_beat import ecg_beat
import numpy as np


class ecg_frame:
    __slots__ = 'beat_array'

    def __init__(self, number_beats):
        self.beat_array = np.empty(number_beats, dtype=object)

    def create_beat(self, beat_data, beat_attribute, index):
        beat = ecg_beat(beat_data, beat_attribute)
        self.beat_array[index] = beat

    def get_beat_array(self):
        return self.beat_array
