import numpy as np
import pandas as pd
import sys
import json

class first_speaker:
    def __init__(self, name, sentences):
        self.name = name
        self.sentences = sentences

class second_speaker:
    def __init__(self, name, sentences):
        self.name = name
        self.sentences = sentences

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(1)
    print("hello")