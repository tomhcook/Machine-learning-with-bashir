import numpy as np
import matplotlib.pyplot as lt
import pandas as pd
def task1import():

#fucntion to import task 3 data
def task3import():
    datas = pd.read_csv('HIV RVG dataset ML.csv')
    print( datas.agg(
        {
            "Alpha":["mean","std","min","max"],
    "Beta":["mean","std","min","max"],
    "Lambda":["mean","std","min","max"],
    "Lambda1":["mean","std","min","max"],
    "Lambda2":["mean","std","min","max"]

        }
    ))
task3import()