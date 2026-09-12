import json
import numpy as np
from data import my_data
with open(r"D:\Torch_frame_work\BPE\mearg.json","r",encoding="utf-8") as f:
    data=json.load(f)
    merg={tuple(map(int,k.split(','))):v for  k,v in data.items()}




def byetes_convertor(text):
    text=bytes(text,encoding="utf-8")
    arr=np.array((list(text)))

    return arr


def pairing(arr):
    p1=arr[:-1]
    p2=arr[1:]
    pair=np.column_stack((p1,p2))

    return pair


def encoder(text):
    i=0
    ids=byetes_convertor(str(text))
    ids=np.concatenate(([276], ids, [277]))
    while len(ids)>=2:
          pair=pairing(ids)
          best_pair=min(pair,key= lambda x :merg.get(tuple(x),float("inf")))
          best_pair=(int(best_pair[0]),int(best_pair[1]))
          if best_pair not in merg:
                break
          
          new_ids=merg[best_pair]

          idss=[]
          i=0

          while i<len(ids):

                if i<len(ids)-1 and ids[i]==best_pair[0] and ids[i+1]==best_pair[1]:

                        idss.append(new_ids)
                        i+=2
                else:

                        idss.append(ids[i])
                        i+=1

          ids=np.array(idss,dtype="int32")
                

    return ids


mx=max(len(byetes_convertor(str(k))) for k in list(my_data()))
def tokenizer(text=None,padd=None,max_length=0,Truncate=None):
      data={}
      if padd is not False:
            ido=encoder(text)
            if max_length!=0 and Truncate is True:
                   zero=np.zeros(max_length,dtype="int32")
                   ido=np.append(ido,zero)
                   ido=ido[:max_length+1]
                   ido=np.concatenate((ido, [277]))

                   data["input_ids"]=ido
                   data["attention_mask"]=np.where(ido!=0,1,0)
                   return ido              
            zero=np.zeros(mx,dtype="int32")
            ido=np.append(ido,zero)
            data["input_ids"]=ido
            data["attention_mask"]=np.where(ido!=0,1,0)
            return ido
                











                    





