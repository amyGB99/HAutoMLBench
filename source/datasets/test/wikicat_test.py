from autogoal.kb import *
from autogoal.ml import AutoML
from dataset import Dataset
from functions_load import *
import utils
from sklearn.metrics import f1_score
import numpy as np 

a = Dataset("wikicat-es", "https://github.com/amyGB99/automl_benchmark/releases/download/wikiann-es/wikiann-es.zip", load_wikicat)
utils.save_dataset_definition(a)
c = utils.load_dataset_definition("wikicat-es")
Xtr,ytr,Xde,yde,Xte,yte = c.loader_func(a)
ytr1= np.asarray(ytr)
yte1= np.asarray(yte)
yde1= np.asarray(yde)
print("kaka")
automl = AutoML(
    input=(Seq[Document], Supervised[VectorCategorical]),
    output= VectorCategorical)
print(len(ytr1))
print(len(Xtr))
print(Xtr[0:5])
# count = 0
# for text in Xtr:
#     print("##########################")
#     print(text)
#     count+=1
#     if count== 5:
#         break
automl.fit(Xtr[0:100],ytr1[0:100])
#print(yte1)

print("aqui")
y = automl.predict(Xte)

#for i in y:  
print(y)
print(automl.best_score_)

#score = automl.score(Xte,yte1)


