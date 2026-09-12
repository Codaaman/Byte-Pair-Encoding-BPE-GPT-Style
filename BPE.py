import json,os
import numpy as np
from collections  import Counter
from data import my_data


def bit_converter(text):
    a=bytes(text,encoding='utf-8')
    arr=np.array(list(a))

    return arr


def pairing(arr):
    p1=arr[:-1]
    p2=arr[1:]

    pair=np.column_stack((p1,p2))

    return pair




def frequency(pair):
    frq=Counter(tuple(row) for row in pair)
    return frq




def mearge(ids,pair,idk):
    new_ids=[]
    i=0
    while i<len(ids):
         
         if i<len(ids)-1 and ids[i]==pair[0] and ids[i+1]==pair[1]:
            new_ids.append(idk)
            i+=2
         else:
             new_ids.append(int(ids[i]))
             i+=1
    return new_ids



for k in list(my_data()):
    vocab=k

k=0
n=260951-256


ids=bit_converter(vocab)

idk=256
vocab_dict={i:chr(i) for i in range(256)}
mearg={}
print("started")
while k<=n:
    par=pairing(ids)
    
    frq=frequency(par)
    a = max(frq, key=frq.get)
    if frq[a]<2:
        break

    values=(int(a[0]),int(a[1]))
    mearg[values]=idk

    first_part = vocab_dict[int(a[0])]
    second_part = vocab_dict[int(a[1])]
    vocab_dict[idk] = first_part + second_part
    new_ids=mearge(ids,(int(a[0]),int(a[1])),idk)
    ids=new_ids
    idk+=1
    k+=1


mearg={ f"{k[0]},{k[1]}": v for k,v in mearg.items()}
with open(r"D:\Torch_frame_work\BPE\mearg.json","w",encoding="utf-8") as f:
    json.dump(mearg,f,indent=4)
    

with open(r"D:\Torch_frame_work\BPE\vocab.json","w",encoding="utf-8") as f:
    json.dump(vocab_dict,f,indent=4)
