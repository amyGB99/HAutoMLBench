

from benchmark import HAutoMLBench
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
# from autogluon.multimodal import MultiModalPredictor
# from autogluon.text import TextPredictor
import pandas as pd 
import json
import os
from benchmark.utils import create_prperties

HAutoMLBench.init()

################## Filter ########################
# print('Benchmark datasets:')
# names = HAutoMLBench.filter()
# print(names)

# print('Binary Classification Datasets')
# bin = HAutoMLBench.filter(task ='binary')
# print(bin)

# print('Multiclass Classification Datasets')
# mult = HAutoMLBench.filter(task ='multiclass')
# print(mult)

# print('Regression Datasets')
# regre = HAutoMLBench.filter(task ='regression')
# print(regre)

# print('Entity Recognition Datasets')
# entity = HAutoMLBench.filter(task ='entity')
# print(entity)

# print('Filter datasets by property columns >=3')
# filter = HAutoMLBench.filter(expresion=('n_columns',3, None))

################### Load ###########################
# print("Load pub-health dataset and display your training set")
# dataset = HAutoMLBench.get_dataset('pub-health')  
# train,test = dataset.loader_func(dataset)
# print(train)

# print('Load Binary Classification datasets and display your training set')
# for name in bin:
#     print("Dataset:" + str(name))
#     dataset = HAutoMLBench.get_dataset(name)  
#     train,test = dataset.loader_func(dataset)
#     print(train)

# print('Load Multiclass Classification datasets and display your training set')
# for name in mult:
#     print("Dataset:" + str(name))
#     dataset = HAutoMLBench.get_dataset(name)  
#     train,test = dataset.loader_func(dataset)
#     print(train)
    
# print('Load Regression datasets and display your training set')
# for name in regre:
#     print("Dataset:" + str(name))
#     dataset = HAutoMLBench.get_dataset(name)  
#     train,test = dataset.loader_func(dataset)
#     print(train)

    
# print('Load Entity Recognition datasets and display your training set')
# for name in entity:
#     print("Dataset:" + str(name))
#     dataset = HAutoMLBench.get_dataset(name)  
#     train,t_train, test, y_tes = dataset.loader_func(dataset)
#     print(train)



############################### Evaluate #################################
# y_true_bin = [1,0,1,1,0]
# y_pred_bin = [0,1,1,1,1]

# y_true_en = [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['O', 'O', 'O'], ['B-NOMBRE_PERSONAL_SANITARIO', 'I-NOMBRE_PERSONAL_SANITARIO', 'I-NOMBRE_PERSONAL_SANITARIO', 'B-CALLE', 'I-CALLE', 'I-CALLE', 'I-CALLE', 'I-CALLE', 'I-CALLE', 'I-CALLE', 'I-CALLE', 'I-CALLE', 'I-CALLE', 'I-CALLE', 'I-CALLE', 'B-TERRITORIO'], ['B-TERRITORIO', 'O', 'B-CORREO_ELECTRONICO']]
# y_pred_en = [['O', 'B-NOMBRE_PERSONAL_SANITARIO', 'O', 'O', 'O', 'O', 'O', 'B-CALLE', 'O', 'O', 'O', 'O', 'O', 'B-CALLE', 'O'], ['O', 'O', 'B-TERRITORIO'], ['0', 'I-NOMBRE_PERSONAL_SANITARIO', 'I-NOMBRE_PERSONAL_SANITARIO', 'B-CALLE', 'I-CALLE', '0', 'I-CALLE', 'I-CALLE', 'I-CALLE', '0', 'I-CALLE', 'I-CALLE', '0', 'B-TERRITORIO', 'I-CALLE', 'I-CALLE'], ['B-TERRITORIO', 'O', 'B-CORREO_ELECTRONICO']]

# y_true_mult = ["Negocios","Economía", "Ingeniería_por_tipo", "Entretenimiento","Negocios", "Economía","Ingeniería_por_tipo", "Entretenimiento"]
# y_pred_mult = ["Negocios","Ingeniería_por_tipo","Economía", "Entretenimiento", "Ingeniería_por_tipo","Ingeniería_por_tipo", "Negocios","Entretenimiento"]

# y_true_reg = [14.709628266086503, 477.47166897541035, 0.6939092986693077, 0.9557622008734076, 0.5016060903771579]
# y_pred_reg = [13.82852241611488, 596.901767909682, 0.7200842528709684, 0.9557622008734076, 0.4435707484313233]


# print('Evaluate Entity Recognition Dataset: meddocan' )
# print(HAutoMLBench.evaluate('meddocan',y_true_en,y_pred_bin))


# print('Evaluate Binary Classification Dataset: wnli-es' )
# print(HAutoMLBench.evaluate('wnli-es',y_true_bin,y_pred_en))

# print('Evaluate Multiclass Classification Dataset: wnli-es' )
# print(HAutoMLBench.evaluate('wikicat-es',y_true_mult, y_pred_mult))

# print('Evaluate Regression Dataset: spanish-wine' )
#print(HAutoMLBench.evaluate('spanish-wine',y_true_reg, y_pred_reg))

######################### Add new Dataset ###########################################
# info_good = {"n_columns": 2, "n_instances": [92720, 23187], "null_values": 0, "targets": "label", "class_labels": None, "positive_class": None, "n_classes": None, "task": "entity", "class_balance": None, "columns_types": {"tokens": "Seqtokens", "tags": "Seqtags"}}

# info_bad = {"n_instances": [80000, 300], "n_columns": 2, "columns_type": {"labels": "text", "text": "text"}, "targets": "labels", "null_values": 0, "task": "multiclass", "classes": 20, "class balance": 1.0}

# def function():
#     pass
#print('Add new dataset without metadata')
#HAutoMLBench.add_dataset('new_dataset','new_url',function)

#print('Add new dataset with metadata')
#HAutoMLBench.add_dataset('new_dataset','new_url',function, metadata= info_good)

#print('Add new dataset with incorrect metadata')
#HAutoMLBench.add_dataset('new_dataset','new_url',function, metadata= info_bad)


############################# Remove Dataset #######################################
#print('Dataset before removing')
#names = HAutoMLBench.filter()
#print(names)

#print("Removing Dataset: wikineural-en")
#HAutoMLBench.remove_dataset('wikineural-en')

#print('Dataset after removing')
#names = HAutoMLBench.filter()
#print(names)
