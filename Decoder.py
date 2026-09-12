import json
import numpy as np
from Eecoder import tokenizer
from data import my_data

with open(r"D:\Torch_frame_work\BPE\vocab.json","r",encoding="utf-8") as f:
    data=json.load(f)
    vocab_dict={int(k):v for k,v in data.items()}
    vocab_dict[276]='<BOS>'
    vocab_dict[277]='<EOF>'


def decoder(text,padd=None):
    if padd is False:
         text=text[~np.isin(text,[276,277])]
    bytes_list=[]
    for i in text:
        words=vocab_dict[i]
        if isinstance(words,str):
            bytes_list.append(words.encode('latin-1'))
        else:
            bytes_list.append(bytes([words]))
    result=b"".join(bytes_list)
    result.decode('utf-8',errors='replace')

    return result.decode('utf-8')






text=[input("plese enter your sentence: ")]

for k in text:
      print(decoder(tokenizer(k),padd=False))




    










