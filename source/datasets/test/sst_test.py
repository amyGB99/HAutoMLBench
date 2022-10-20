from autogoal.kb import *
from autogoal.ml import AutoML
from dataset import Dataset
from functions_load import *
import utils
from sklearn.metrics import f1_score
import numpy as np 

a = Dataset("sst-en", "https://github.com/amyGB99/automl_benchmark/releases/download/paws-x/paws-x-es.zip", load_sst_en )
c = utils.load_dataset_definition("sst-en")
Xtr,ytr,Xde,yde,Xte,yte = c.loader_func(a)
ytr1= np.asarray(ytr)
yte1= np.asarray(yte)
yde1= np.asarray(yde)
print("kaka")
automl = AutoML(
        input=(Seq[Sentence], Supervised[VectorCategorical]),
        output= VectorCategorical)

print(len(ytr1))
automl.fit(Xtr,ytr1)
print(yte1)

print("aqui")
y = automl.predict(Xte)

#for i in y:  
print(y)
print(automl.best_score_)

#score = automl.score(Xte,yte1)


