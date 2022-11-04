

from benchmark import AutoMLBench
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# from autogluon.multimodal import MultiModalPredictor
# from autogluon.text import TextPredictor
import pandas as pd 
import json
import os
#AutoMLBench.create_datasets()
# dict_ = {}
# names  =AutoMLBench.names
# print(names)
#print(names)
#info = {"n_instances": [80000, 300], "n_columns": 2, "columns_type": {"labels": "text", "text": "text"}, "targets": ["labels"], "null_values": 0, "task": "multiclass", "classes": 20, "class balance": 1.0}
#AutoMLBench.new_dataset('wikineural-en','llaala','f',info = info)
#AutoMLBench.remove_dataset('wikineural-en')

#AutoMLBench.new_dataset('holi','aqui estoy','f')
#AutoMLBench.new_dataset('holi','aqui','f',info = info)
#names  =AutoMLBench.names
#print(names)
#print(AutoMLBench.info['holi'])
#AutoMLBench.remove_dataset('holi',permanent=True)
#names  =AutoMLBench.names
#print(names)
#print(AutoMLBench.filter())
#print(AutoMLBench.names)
local_path = os.path.dirname(os.path.realpath(__file__))
result_path = os.path.join(local_path,'y_gpu.json')
with open(result_path,'r') as fp:
    results = json.load(fp)
y_pred = results['stsb-en']    
#y_pred = ["positive", "positive", "positive", "positive", "positive", "positive", "negative", "positive", "negative", "positive", "negative", "positive", "negative", "negative", "negative", "positive", "positive", "negative", "positive", "positive", "positive", "positive", "positive", "negative", "positive", "positive", "positive", "negative", "positive", "positive", "positive", "positive", "positive", "positive", "positive", "positive", "positive", "negative", "positive", "positive", "negative", "negative", "positive", "positive", "positive", "positive", "positive", "negative", "negative", "positive", "positive", "positive", "negative", "positive", "positive", "positive", "positive", "positive", "positive", "negative", "positive", "negative", "negative", "positive", "negative", "positive", "negative", "negative", "positive", "positive", "positive", "positive", "negative", "positive", "positive", "negative", "positive", "positive", "negative", "negative", "positive", "positive", "negative", "positive", "positive", "positive", "positive", "positive", "negative", "negative", "positive", "negative", "negative", "positive", "positive", "positive", "negative", "positive", "positive", "negative", "positive", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "positive", "negative", "positive", "negative", "positive", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "positive", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "positive", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "positive", "negative", "positive", "negative", "positive", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "positive", "negative", "negative", "negative", "negative", "negative"]
#X_train, y_train, X_test, y_test = AutoMLBench.load_dataset("sentiment-lexicons-es",format='pandas',in_xy=True,samples=2)
X_train, y_train, X_test, y_test = AutoMLBench.load_dataset("stsb-en",format='pandas',in_xy=True,samples=2)

y = y_test['score'].tolist()
#label = ['negative','positive']
#AutoMLBench.evaluate('sentiment-lexicons-es',y,y_pred,'binary','positive',label)
AutoMLBench.evaluate('stsb-en',y,y_pred,'regression')

#print(y)
#print(AutoMLBench.scoring['f1_score_w'][0](y_true = y,y_pred = y_pred, labels =label,average =AutoMLBench.scoring['f1_score_w'][1] ))
#f1_ = Metric.f1()
#print(f1)
#print(f1_(y_true= y,y_pred))
#X_train, y_train, X_test, y_test = AutoMLBench.load_dataset("wikiann-es",format='list',in_xy=True,samples=2)
# print(len(X_train))
# print(len(X_test))


def create_columns_type(names):
    all_types = {}
    for dataset in names:
        print('#####################################################')
        print(dataset)
        types = {}
        if dataset == "wikiann-es" or dataset =='meddocan' or dataset =='wikineural-es' or dataset =='wikineural-en':
            all_types[dataset] = {'tokens':'Seqtokens','tags':'Seqtags'}
            continue
        target = AutoMLBench.info[dataset]['properties']['targets'][0]
        all = AutoMLBench.load_dataset(dataset,format='pandas',in_xy=False,samples=1)
        print(all.loc[1])
        for column in all.columns:
            t = all[column].dtype.name
            if t== 'int64':
                types[column] = 'int'
            elif t== 'float64':
                types[column] = 'float'  
            elif t== 'object':
                types[column] = 'text' 
            else:
                types[column] = t       
            #print(type(all[column].dtype.name))
            #['sentence1','sentence2','text','Title','Review Text','location','company_profile','description','requirements','benefits','country','']
        all_types[dataset] = types
    with open('columns_types.json', 'w') as fp:
        json.dump(all_types, fp,indent= 4)     




def create_prperties(names):
    dict_ ={}
    for dataset in names:
        print(dataset)
        if dataset == "wikiann-es" or dataset =='meddocan' or dataset =='wikineural-es' or dataset =='wikineural-en':
            continue
        else:
            train,test = AutoMLBench.load_dataset(dataset,format='pandas',in_xy=False,samples=2)
            post = None
            
            columns = len(train.columns)
            intances = [len(train.axes[0]),len(test.axes[0])]
            count_null = int(all.isnull().sum().sum())
            if AutoMLBench.info[dataset]['classes'] == None:
                number_class = None
                balance = None
            else:    
                number_class = len(all[target].unique())
                number_clases_train = train[target].value_counts()
                min = number_clases_train.min()
                max = number_clases_train.max()
                balance = min/max 
            dict_[dataset] = {'n_columns':columns,'n_instances':intances,'null_values':count_null,'post_label': post,'classes': number_class,'balance':balance}
    
    with open('datos.json', 'w') as fp:
        json.dump(dict_, fp,indent= 4)  
names = AutoMLBench.filter(task ='binary', expresion=('n_columns',3,None))

#train,y_train, test ,y_test= AutoMLBench.load_dataset("stsb-en",format='list',in_xy=True,samples=2)
#train,y_train, test ,y_test= AutoMLBench.load_dataset("stsb-es",in_xy=True,samples=2)
#train,y_train, test ,y_test = AutoMLBench.load_dataset("predict-salary",format='list',in_xy=True,samples=2)
#train,test = AutoMLBench.load_dataset("women-clothing",format='pandas',in_xy=False,samples=2)
#AutoMLBench.load_dataset("fraudulent-jobs",format='pandas',in_xy=False,samples=2)
#train,test = AutoMLBench.load_dataset("inferes",format='pandas',in_xy=False,samples=2)
#train,ytr, test,yte = AutoMLBench.load_dataset("spanish-wine",format='list',in_xy=True,samples=2)


#train, test,  = AutoMLBench.load_dataset("vaccine-en",format='pandas',in_xy=False,samples=2)
#train, test,  = AutoMLBench.load_dataset("language-identification",format='pandas',in_xy=False,samples=2)
#train, test,  = AutoMLBench.load_dataset("twiter-human-bots",format='pandas',in_xy=False,samples=2)
#train,test = AutoMLBench.load_dataset("sentiment-lexicons-es",format='pandas',in_xy=False,samples=2)
#all,y = AutoMLBench.load_dataset("sentiment-lexicons-es",format='pandas',in_xy=True,samples=1)

#X_train, y_train, X_test, y_test = AutoMLBench.load_dataset("meddocan",format='list',in_xy=True,samples=2)

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