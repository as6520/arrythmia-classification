"""
File: ecg_beat.py
Author: Ameya Shringi(as6520@g.rit.edu)
        Vishal Garg
Description: Data structure to store a single heart beat
             and its attribute
"""
class ecg_beat:

    __slots__ = "lead", "attribute"

    def __init__(self, lead, attribute):
        """
        Constructor for the class
        :param lead: data array of ecg signal
        :param attribute: attribute associated with the signal
        :return: None
        """
        self.lead = lead
        self.attribute = attribute

    def get_lead(self):
        """
        Getter function to get data
        :return: return the data
        """
        return self.lead

    def get_attribute(self):
        """
        Getter function to get attribute
        :return: return the attribute
        """
        return self.attribute
