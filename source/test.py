from numpy import dtype
from benchmark import AutoMLBench
# from autogluon.multimodal import MultiModalPredictor
# from autogluon.text import TextPredictor
import pandas as pd 
import json
AutoMLBench.init()
dict_ = {}
names  =AutoMLBench.names
#print(names)
print(AutoMLBench.filter(task ='binary',expresion=None))
all_types = {}
# for dataset in names:
#     print('#####################################################')
#     print(dataset)
#     types = {}
#     if dataset == "wikiann-es" or dataset =='meddocan' or dataset =='wikineural-es' or dataset =='wikineural-en':
#         all_types[dataset] = {'tokens':'Seqtokens','tags':'Seqtags'}
#         continue
#     target = AutoMLBench.info[dataset]['properties']['targets'][0]
#     all = AutoMLBench.load_dataset(dataset,format='pandas',in_xy=False,samples=1)
#     print(all.loc[1])
#     for column in all.columns:
#         t = all[column].dtype.name
#         if t== 'int64':
#             types[column] = 'int'
#         elif t== 'float64':
#             types[column] = 'float'  
#         elif t== 'object':
#             types[column] = 'text' 
#         else:
#             types[column] = t       
#         #print(type(all[column].dtype.name))
#         #['sentence1','sentence2','text','Title','Review Text','location','company_profile','description','requirements','benefits','country','']
#     all_types[dataset] = types
# with open('columns_types.json', 'w') as fp:
#     json.dump(all_types, fp,indent= 4)     

# def create_prperties():
#     dict_ ={}
#     for dataset in names:
#         print(dataset)
#         if dataset == "wikiann-es" or dataset =='meddocan' or dataset =='wikineural-es' or dataset =='wikineural-en':
#             continue
#         else:
#             train,test = AutoMLBench.load_dataset(dataset,format='pandas',in_xy=False,samples=2)
#             columns = len(train.columns)
#             intances = [len(train.axes[0]),len(test.axes[0])]
#             count_null = int(all.isnull().sum().sum())
#             if AutoMLBench.info[dataset]['classes'] == None:
#                 number_class = None
#                 balance = None
#             else:    
#                 number_class = len(all[target].unique())
#                 number_clases_train = train[target].value_counts()
#                 min = number_clases_train.min()
#                 max = number_clases_train.max()
#                 balance = min/max 
#             dict_[dataset] = {'n_columns':columns,'n_instances':intances,'null_values':count_null,'classes': number_class, 'balance':balance}
    
#     with open('datos_a.json', 'w') as fp:
#         json.dump(dict_, fp,indent= 4)  

#train,y_train, test ,y_test= AutoMLBench.load_dataset("stsb-en",format='list',in_xy=True,samples=2)
#train,y_train, test ,y_test= AutoMLBench.load_dataset("stsb-es",in_xy=True,samples=2)
#train,y_train, test ,y_test = AutoMLBench.load_dataset("predict-salary",format='list',in_xy=True,samples=2)
#train,test = AutoMLBench.load_dataset("women-clothing",format='pandas',in_xy=False,samples=2)
#AutoMLBench.load_dataset("fraudulent-jobs",format='pandas',in_xy=False,samples=2)
#train,test = AutoMLBench.load_dataset("inferes",format='pandas',in_xy=False,samples=2)
#train,ytr, test,yte = AutoMLBench.load_dataset("spanish-wine",format='list',in_xy=True,samples=2)
#X_train, y_train, X_test, y_test = AutoMLBench.load_dataset("meddocan",format='pandas',in_xy=False,samples=2)
#train, test,  = AutoMLBench.load_dataset("vaccine-en",format='pandas',in_xy=False,samples=2)
#train, test,  = AutoMLBench.load_dataset("language-identification",format='pandas',in_xy=False,samples=2)
#train, test,  = AutoMLBench.load_dataset("twiter-human-bots",format='pandas',in_xy=False,samples=2)
#train,test = AutoMLBench.load_dataset("sentiment-lexicons-es",format='pandas',in_xy=False,samples=2)
#all,y = AutoMLBench.load_dataset("sentiment-lexicons-es",format='pandas',in_xy=True,samples=1)


#train,y_train, test ,y_test = AutoMLBench.load_dataset("sst-en", in_xy = True, samples = 2)

#train = AutoMLBench.load_dataset("sst-en", in_xy = False, samples = 1)

#print(train)
# train_data,test_data = AutoMLBench.load_dataset("paws-x-en",format='pandas',in_xy=False,samples=2)
# print("fit")
# time_limit = 10*60
# predictor = TextPredictor(label='label').fit(train_data,time_limit=time_limit)
# print("predict")
# predictions = predictor.predict(test_data)
# print("evaluate")
# score = predictor.evaluate(test_data)
# print(score)