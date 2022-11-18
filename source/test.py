

from benchmark import HAutoMLBench
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
# from autogluon.multimodal import MultiModalPredictor
# from autogluon.text import TextPredictor
import pandas as pd 
import json
import os
from benchmark.utils import create_prperties

HAutoMLBench.init()

print('Benchmark datasets:')
#names = HAutoMLBench.filter()
#print(names)
dataset = HAutoMLBench.get_dataset('pub-health')  
#train,test = dataset.loader_func(dataset)
print("holi")
#print(test.iloc[0,7])
path = '/media/amanda/DATA1/School/Thesis/implementation/benchmark actual/automl_benchmark/source/benchmark/columns_types.json'
path2 = '/media/amanda/DATA1/School/Thesis/implementation/benchmark actual/automl_benchmark/source/benchmark/columns_types2.json'
# with open(path2, 'r') as fp:
#   all_types = json.load(fp)
#   #del all_types['vaccine-en']
#   all_types['pub-health'] = {'claim': 'text', 'date_published': 'datetime','explanation': 'text','fact_checkers': 'text', 'main_text': 'text','sources' : 'text','label': 'category',  'subjects': 'text'}

# with open(path, 'w') as fp:
#   json.dump(all_types, fp,indent= 4)
# print('Binary Classification Datasets')
# bin = HAutoMLBench.filter(task ='binary')
# print(bin)

# print('Multiclass Classification Datasets')
# mult = HAutoMLBench.filter(task ='multiclass')
# print(mult)
# dataset = HAutoMLBench.get_dataset('pub-health')  
# train,test = dataset.loader_func(dataset)
# print(test)

# for name in names:
#   dataset = HAutoMLBench.get_dataset(name)  
#   ret = dataset.loader_func(dataset)
#   print(ret)

#create_prperties()

#print(y_te)

# print('Regression Datasets')
# regre = HAutoMLBench.filter(task ='regression')
# #print(regre)
# dict_ = {}
#for reg in regre:
#   info = HAutoMLBench.load_info(reg)  
#   train,y_tr,test, y_te  = HAutoMLBench.load_dataset(reg,format='list',in_xy= True,samples=2,target = info['targets'])
#   dict_[reg] = y_te
# print(dict_.keys())
# with open('/media/amanda/DATA1/School/Thesis/implementation/benchmark actual/automl_benchmark/source/y.json', 'w') as fp:
#         json.dump(dict_, fp,indent= 4,ensure_ascii= False) 
#print('Entity Recognition Datasets')
#entity = HAutoMLBench.filter(task ='entity')
# #print(entity)
# y_true = [3,0,2,2,1]
# y_predic = [5,2,1,1,1]

# y_true = [[0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 3, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 6, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
# y_predict =[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 6, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] 

# y_true = [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['O', 'O', 'O'], ['B-NOMBRE_PERSONAL_SANITARIO', 'I-NOMBRE_PERSONAL_SANITARIO', 'I-NOMBRE_PERSONAL_SANITARIO', 'B-CALLE', 'I-CALLE', 'I-CALLE', 'I-CALLE', 'I-CALLE', 'I-CALLE', 'I-CALLE', 'I-CALLE', 'I-CALLE', 'I-CALLE', 'I-CALLE', 'I-CALLE', 'B-TERRITORIO'], ['B-TERRITORIO', 'O', 'B-CORREO_ELECTRONICO']]
# y_predict = [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['O', 'O', 'O'], ['B-NOMBRE_PERSONAL_SANITARIO', 'I-NOMBRE_PERSONAL_SANITARIO', 'I-NOMBRE_PERSONAL_SANITARIO', 'B-CALLE', 'I-CALLE', 'I-CALLE', 'I-CALLE', 'I-CALLE', 'I-CALLE', 'I-CALLE', 'I-CALLE', 'I-CALLE', 'I-CALLE', 'I-CALLE', 'I-CALLE', 'B-TERRITORIO'], ['B-TERRITORIO', 'O', 'B-CORREO_ELECTRONICO']]


# y_true = ["Negocios",
#             "Economía",
#             "Ingeniería_por_tipo",
#             "Entretenimiento","Negocios",
#             "Economía",
#             "Ingeniería_por_tipo",
#             "Entretenimiento"]

# y_predic = ["Negocios",
#             "Ingeniería_por_tipo","Economía","Entretenimiento","Ingeniería_por_tipo","Ingeniería_por_tipo","Negocios","Entretenimiento"]

#print(HAutoMLBench.evaluate('meddocan',y_true,y_predict))

#print(HAutoMLBench.evaluate('wikicat-es',y_true, y_predic))
# print('Filter datasets by property columns >=3')
# filter = HAutoMLBench.filter(expresion=('n_columns',3,None))
# print(filter)
# info = HAutoMLBench.load_info('google-guest')
# print(len( info['targets']))
# train,y_tr,test, y_te  = HAutoMLBench.load_dataset('google-guest',format='list',in_xy= True,samples=2,target = info['targets'])
# print(y_tr)
# print('Filter datasets by property task = binary and  columns >=3')
# filter = HAutoMLBench.filter(task = 'binary',expresion=('n_columns',3,None))
# print(filter)

##### Add new Dataset 
#info_good = {"n_columns": 2, "n_instances": [92720, '23187'], "null_values": 0, "targets": "label", "class_labels": None, "positive_class": None, "n_classes": None, "task": "entity", "class_balance": None, "columns_types": {"tokens": "Seqtokens", "tags": "Seqtags"}}


#info_bad = {"n_instances": [80000, 300], "n_columns": 2, "columns_type": {"labels": "text", "text": "text"}, "targets": "labels", "null_values": 0, "task": "multiclass", "classes": 20, "class balance": 1.0}
def f():
    pass
#HAutoMLBench.add_dataset('holi','aqui estoy',f)
#HAutoMLBench.add_dataset('holi','aqui',f, metadata= info_good)
#names  =HAutoMLBench.names
#print(names)
#print(HAutoMLBench.info['holi'])
#print("Remove Dataset")
#HAutoMLBench.remove_dataset('wikineural-en')
#HAutoMLBench.remove_dataset('holi')

#names  =HAutoMLBench.names
#print(names)
# print(names)
# for name in names:
#   if name != 'wikiann-es' and name != 'meddocan' and name !='wikineural-es' and name != 'wikineural-en':      
#     print("#############################################")
#     print(name)
#     train, test  = HAutoMLBench.load_dataset(name,format='pandas',in_xy=False,samples=2)
#     print(len(train))
#     print(train.loc[1])
#     print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
#     print(len(test))
#     print(test.loc[1])
    
     

#create_prperties()
min_fiv_infe = [14.709628266086503, 477.47166897541035, 0.6939092986693077, 0.9557622008734076, 0.5016060903771579]
min_five_colum = [13.82852241611488, 596.901767909682, 0.7200842528709684, 0.9557622008734076, 0.4435707484313233]

min_fiften = [9.922474660971805, 599.679312598897, 0.6635479766290717, 0.9506574634447019, 0.4489532572135888]

regression= ['spanish-wine', 'price-book', 'stsb-en', 'stsb-es', 'google-guest']