import pandas as pd
import numpy as np

def my_data():
    l=[]
    df=pd.read_csv(r"D:\Torch_frame_work\BPE\corpus.csv")
    for k in df["text"]:
        yield k