from typing import List, Tuple
from autogoal.kb import *
from autogoal.ml import AutoML
from dataset import Dataset
from functions_load import *
import utils
from sklearn.metrics import f1_score
import numpy as np 
from autogoal.search import (
     RichLogger,
     PESearch,
 )
a = Dataset("paws-x-es", "https://github.com/autogoal/benchmark/releases/download/paws-x/paws-x-en.zip", load_two_sentences)
#a = Dataset("sst-en", "https://github.com/amyGB99/automl_benchmark/releases/download/paws-x/paws-x-es.zip", load_sst_en )
utils.save_dataset_definition(a)
d = utils.load_dataset_definition("paws-x-es")
X_train,y_train,X_test,y_test = d.loader_func(a,format = "list")
ytr1= np.asarray(y_train)
#yte1= np.asarray(y_test)
#print("kaka")
automl = AutoML(
        input =(Seq[Seq[Sentence]], Supervised[VectorCategorical]),
        output = VectorCategorical,score_metric= f1_score,cross_validation_steps=1,)
loggers = [RichLogger()]
print("fit")
print(type(X_train[0]))
#print(len(X_train))
automl.fit(X_train,ytr1,logger= loggers)
#print(yte1)

print("predict")
#y = automl.predict(X_test)
#print(y)

#print(automl.best_score_)
#score = automl.score(X_test,yte1)

# from autogoal.ml import AutoML
# from autogoal.datasets import meddocan
# from autogoal.search import (
#     RichLogger,
#     PESearch,
# )
# from autogoal.kb import *

# from autogoal.contrib import find_classes

# # ## Experimentation

# # Instantiate the classifier.
# # Note that the input and output types here are defined to match the problem statement,
# # i.e., entity recognition.

# classifier = AutoML(
#     search_algorithm=PESearch,
#     input=(Seq[Seq[Word]], Supervised[Seq[Seq[Label]]]),
#     output=Seq[Seq[Label]],
#     score_metric=meddocan.F1_beta,
#     cross_validation_steps=1,
#     # Since we only want to try neural networks, we restrict
#     # the contrib registry to algorithms matching with `Keras`.
#     registry=find_classes("Keras|Bert"),
#     # We need to give some extra time because neural networks are slow
#     evaluation_timeout=300,
#     search_timeout=1800,
# )

# # Basic logging configuration.

# loggers = [RichLogger()]

# # Finally, loading the MEDDOCAN dataset, running the `AutoML` instance,
# # and printing the results.

# X_train, y_train, X_test, y_test = meddocan.load()
# print(y_test)
# #classifier.fit(X_train, y_train, logger=loggers)
# #score = classifier.score(X_test, y_test)

# #print(score)

