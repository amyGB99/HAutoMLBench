
from autogoal.ml import AutoML
#from autogoal.search import RichLogger
from autogoal.kb import *
from dataset import Dataset
from functions_load import load_paws_en, load_paws_es
import utils
import numpy as np


a = Dataset("paws-x-en", "https://github.com/amyGB99/automl_benchmark/releases/download/paws-x/paws-x-es.zip", load_paws_en )
c = utils.load_dataset_definition("paws-x-en")
Xtr,ytr,Xde,yde,Xte,yte = c.loader_func(a)
ytr1= np.asarray(ytr)
yte1= np.asarray(yte)
yde1= np.asarray(yde)
#print(Xtr[0:10])
automl = AutoML(
    input=(Seq[Seq[Sentence]], Supervised[VectorCategorical]),
    output= VectorCategorical)
automl.fit(Xtr[0:50],ytr1[0:50])
print(automl.best_score_)
#from autogoal.datasets import haha
# from autogoal.kb import (MatrixContinuousDense, 
#                          Supervised, 
#                          VectorCategorical)
# from autogoal.ml import AutoML

# # Load dataset
#X_train, y_train, X_test, y_test= haha.load()


# # Instantiate AutoML and define input/output types
# #automl = AutoML(
#    # input=(MatrixContinuousDense, 
#     #       Supervised[VectorCategorical]),
#     #output=VectorCategorical
# #)

# # Run the pipeline search process
# #automl.fit(X, y)

# # Report the best pipeline
# #print(automl.best_pipeline_)
# #print(automl.best_score_)
