from autogoal.kb import (Sentence, 
                         Supervised, 
                         VectorCategorical)
from autogoal.ml import AutoML
from dataset import Dataset
from functions_load import *
import utils


a = Dataset("sst-en", "https://github.com/amyGB99/automl_benchmark/releases/download/paws-x/paws-x-es.zip", load_sst_en )
c = utils.load_dataset_definition("sst-en")
c.loader_func(a)
automl = AutoML(
    input=(List(List(Sentence)), Supervised[VectorCategorical]),
    output= VectorCategorical)


