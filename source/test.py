

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
#names = AutoMLBench.filter()
#print(names)

#info = {"n_instances": [80000, 300], "n_columns": 2, "columns_type": {"labels": "text", "text": "text"}, "targets": ["labels"], "null_values": 0, "task": "multiclass", "classes": 20, "class balance": 1.0}
#AutoMLBench.new_dataset('wikineural-en','llaala','f',info = info)
#AutoMLBench.remove_dataset('wikineural-en')

#AutoMLBench.new_dataset('holi','aqui estoy','f')
#AutoMLBench.new_dataset('holi','aqui','f',info = info)
#names  =AutoMLBench.names
#print(names)
#print(AutoMLBench.info['holi'])
AutoMLBench.remove_dataset('holi')
#names  =AutoMLBench.names
#print(names)
print(AutoMLBench.filter())
#print(AutoMLBench.names)
#local_path = os.path.dirname(os.path.realpath(__file__))
#result_path = os.path.join(local_path,'y_gpu.json')
#with open(result_path,'r') as fp:
 #   results = json.load(fp)
#y_pred = results['stsb-en']    
#y_pred = ["positive", "positive", "positive", "positive", "positive", "positive", "negative", "positive", "negative", "positive", "negative", "positive", "negative", "negative", "negative", "positive", "positive", "negative", "positive", "positive", "positive", "positive", "positive", "negative", "positive", "positive", "positive", "negative", "positive", "positive", "positive", "positive", "positive", "positive", "positive", "positive", "positive", "negative", "positive", "positive", "negative", "negative", "positive", "positive", "positive", "positive", "positive", "negative", "negative", "positive", "positive", "positive", "negative", "positive", "positive", "positive", "positive", "positive", "positive", "negative", "positive", "negative", "negative", "positive", "negative", "positive", "negative", "negative", "positive", "positive", "positive", "positive", "negative", "positive", "positive", "negative", "positive", "positive", "negative", "negative", "positive", "positive", "negative", "positive", "positive", "positive", "positive", "positive", "negative", "negative", "positive", "negative", "negative", "positive", "positive", "positive", "negative", "positive", "positive", "negative", "positive", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "positive", "negative", "positive", "negative", "positive", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "positive", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "positive", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "positive", "negative", "positive", "negative", "positive", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "positive", "negative", "negative", "negative", "negative", "negative"]
#X_train, y_train, X_test, y_test = AutoMLBench.load_dataset("sentiment-lexicons-es",format='pandas',in_xy=True,samples=2)
#X_train, y_train, X_test, y_test = AutoMLBench.load_dataset("stsb-en",format='pandas',in_xy=True,samples=2)

#y = y_test['score'].tolist()
#label = ['negative','positive']
#AutoMLBench.evaluate('sentiment-lexicons-es',y,y_pred,'binary','positive',label)
#@AutoMLBench.evaluate('stsb-en',y,y_pred,'regression')

#print(y)
#print(AutoMLBench.scoring['f1_score_w'][0](y_true = y,y_pred = y_pred, labels =label,average =AutoMLBench.scoring['f1_score_w'][1] ))
#f1_ = Metric.f1()
#print(f1)
#print(f1_(y_true= y,y_pred))
#X_train, y_train, X_test, y_test = AutoMLBench.load_dataset("wikiann-es",format='list',in_xy=True,samples=2)
# print(len(X_train))
# print(len(X_test))





def create_columns_type():
    labels  = [['label'], ['label'], ['label'], ['label'], ['label'],['label'],['stroke'],['Class Name'],['fraudulent'],['price'] ,['final_status'] ,['Price'], ['Label'],['salary'],['score'],['score'],['is_humor','average'],['label'],['label'],['label'],['label'],['label'],['label'],['labels'],['account_type'],['question_asker_intent_understanding','question_body_critical','question_conversational','question_expect_short_answer','question_fact_seeking','question_has_commonly_accepted_answer','question_interestingness_others','question_interestingness_self','question_multi_intent','question_not_really_a_question','question_opinion_seeking','question_type_choice','question_type_compare','question_type_consequence','question_type_definition','question_type_entity','question_type_instructions','question_type_procedure','question_type_reason_explanation','question_type_spelling','question_well_written','answer_helpful', 'answer_level_of_information','answer_plausible','answer_relevance','answer_satisfaction','answer_type_instructions','answer_type_procedure','answer_type_reason_explanation', 'answer_well_written']]
    names =["paws-x-en","paws-x-es","wnli-es","wikiann-es","wikicat-es","sst-en",
                    "stroke-prediction","women-clothing","fraudulent-jobs","spanish-wine",
                    "project-kickstarter","price-book","inferes","predict-salary","stsb-en",
                    "stsb-es","haha", "meddocan","vaccine-es","vaccine-en","sentiment-lexicons-es",
                    "wikineural-en","wikineural-es","language-identification","twitter-human-bots","google-guest"]
    all_types = {}
    for dataset,i in zip(names,range(len(names))):
        print('#####################################################')
        print(dataset)
        types = {}
        if dataset == "wikiann-es" or dataset =='meddocan' or dataset =='wikineural-es' or dataset =='wikineural-en':
            all_types[dataset] = {'tokens':'Seqtokens','tags':'Seqtags'}
            continue
        target = labels[i]
        all = AutoMLBench.load_dataset(dataset,format='pandas',in_xy=False,samples=1)
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
    with open('/media/amanda/DATA1/School/Thesis/implementation/benchmark actual/automl_benchmark/source/benchmark/columns_types.json', 'w') as fp:
        json.dump(all_types, fp,indent= 4) 
        
def create_prperties():
    labels  = [['label'], ['label'], ['label'], ['label'], ['label'],['label'],['stroke'],['Class Name'],['fraudulent'],['price'] ,['final_status'] ,['Price'], ['Label'],['salary'],['score'],['score'],['is_humor','average'],['label'],['label'],['label'],['label'],['label'],['label'],['labels'],['account_type'],['question_asker_intent_understanding','question_body_critical','question_conversational','question_expect_short_answer','question_fact_seeking','question_has_commonly_accepted_answer','question_interestingness_others','question_interestingness_self','question_multi_intent','question_not_really_a_question','question_opinion_seeking','question_type_choice','question_type_compare','question_type_consequence','question_type_definition','question_type_entity','question_type_instructions','question_type_procedure','question_type_reason_explanation','question_type_spelling','question_well_written','answer_helpful', 'answer_level_of_information','answer_plausible','answer_relevance','answer_satisfaction','answer_type_instructions','answer_type_procedure','answer_type_reason_explanation', 'answer_well_written']]
    regres =['spanish-wine', 'price-book', 'stsb-en', 'stsb-es', 'google-guest']
    names = ["paws-x-en","paws-x-es","wnli-es","wikiann-es","wikicat-es","sst-en",
                    "stroke-prediction","women-clothing","fraudulent-jobs","spanish-wine",
                    "project-kickstarter","price-book","inferes","predict-salary","stsb-en",
                    "stsb-es","haha", "meddocan","vaccine-es","vaccine-en","sentiment-lexicons-es",
                    "wikineural-en","wikineural-es","language-identification","twitter-human-bots","google-guest"]
    dict_ ={}
    for dataset,i in zip(names,range(len(names))):
        print(dataset)
        if dataset == "wikiann-es" or dataset =='meddocan' or dataset =='wikineural-es' or dataset =='wikineural-en':
            if dataset == "wikiann-es" or dataset =='meddocan':
              train,ytra,test, ytes = AutoMLBench.load_dataset(dataset,format='pandas',in_xy=False,samples=1)
              columns = 2
              instances = [len(train),len(test)]
              count_null = 0
              number_class = None
              balance = None
            else:  
              continue
        else:
            train,test = AutoMLBench.load_dataset(dataset,format='pandas',in_xy=False,samples=2)
            all = AutoMLBench.load_dataset(dataset,format='pandas',in_xy=False,samples=1)

            target = labels[i]

            columns = len(train.columns)
            instances = [len(train.axes[0]),len(test.axes[0])]
            count_null = int(all.isnull().sum().sum())
            if dataset in regres:
              print('kaka')
              number_class = None
              balance = None
              if isinstance(target,str): 
                count_null_class = all[target].isnull().sum()
              else:  
                count_null_class = all[target[0]].isnull().sum()
              print(count_null_class)
            else:
              if isinstance(target,str):    
                number_class = len(all[target].unique())
                number_clases_train = train[target].value_counts()
                count_null_class = all[target].isnull().sum()
              else: 
                number_class = len(all[target[0]].unique())
                number_clases_train = train[target[0]].value_counts() 
                count_null_class = all[target[0]].isnull().sum()
              print(count_null_class)
              min = number_clases_train.min()
              max = number_clases_train.max()
              balance = min/max 
        dict_[dataset] = {'n_columns':columns,'n_instances':instances,'null_values':count_null,'classes': number_class,'balance':balance}
    
    with open('/media/amanda/DATA1/School/Thesis/implementation/benchmark actual/automl_benchmark/source/benchmark/properties.json', 'w') as fp:
        json.dump(dict_, fp,indent= 4)    

#create_columns_type()
#names = AutoMLBench.filter(task ='binary', expresion=('n_columns',3,None))

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