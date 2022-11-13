

from benchmark import HAutoMLBench
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# from autogluon.multimodal import MultiModalPredictor
# from autogluon.text import TextPredictor
import pandas as pd 
import json
import os

#HAutoMLBench.create_datasets()

# print('Benchmark datasets:')
# names = HAutoMLBench.filter()
# print(names)

# print('Binary Classification Datasets')
#bin = HAutoMLBench.filter(task ='binary')
#print(bin)

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
# filter = HAutoMLBench.filter(expresion=('n_columns',3,None))
# print(filter)
# info = HAutoMLBench.load_info('google-guest')
# print(len( info['targets']))
# train,y_tr,test, y_te  = HAutoMLBench.load_dataset('google-guest',format='list',in_xy= True,samples=2,target = info['targets'])
# print(y_tr)
# print('Filter datasets by property task = binary and  columns >=3')
# filter = HAutoMLBench.filter(task = 'binary',expresion=('n_columns',3,None))
# print(filter)

#info = {"n_instances": [80000, 300], "n_columns": 2, "columns_type": {"labels": "text", "text": "text"}, "targets": ["labels"], "null_values": 0, "task": "multiclass", "classes": 20, "class balance": 1.0}
#HAutoMLBench.new_dataset('wikineural-en','llaala','f',info = info)
#HAutoMLBench.remove_dataset('holi')

#HAutoMLBench.new_dataset('holi','aqui estoy','f')
#HAutoMLBench.new_dataset('holi','aqui','f',info = info)
#names  =HAutoMLBench.names
#print(names)
#print(HAutoMLBench.info['holi'])
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
        all = HAutoMLBench.load_dataset(dataset,format='pandas',in_xy=False,samples=1)
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
    labels  = [['label'], ['label'], ['label'], ['label'], ['label'],['label'],['stroke'],['Class Name'],['fraudulent'],['price'] ,['final_status'] ,['Price'], ['Label'],['salary'],['score'],['score'],['is_humor'],['label'],['label'],['label'],['label'],['label'],['label'],['labels'],['account_type'],['question_asker_intent_understanding','question_body_critical','question_conversational','question_expect_short_answer','question_fact_seeking','question_has_commonly_accepted_answer','question_interestingness_others','question_interestingness_self','question_multi_intent','question_not_really_a_question','question_opinion_seeking','question_type_choice','question_type_compare','question_type_consequence','question_type_definition','question_type_entity','question_type_instructions','question_type_procedure','question_type_reason_explanation','question_type_spelling','question_well_written','answer_helpful', 'answer_level_of_information','answer_plausible','answer_relevance','answer_satisfaction','answer_type_instructions','answer_type_procedure','answer_type_reason_explanation', 'answer_well_written']]
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
          train,ytra,test, ytes = HAutoMLBench.load_dataset(dataset,format='pandas',in_xy=False,samples=2)
          columns = 2
          instances = [len(train),len(test)]
          count_null = 0
          number_class = None
          balance = None
          labels_d = None
          task = 'entity'
          pos = None
          
        else:
            train,test = HAutoMLBench.load_dataset(dataset,format='pandas',in_xy=False,samples=2)
            all = HAutoMLBench.load_dataset(dataset,format='pandas',in_xy=False,samples=1)
            if dataset != "google-guest":
              target = labels[i][0]
            else:
              target = labels[i]

            columns = len(train.columns)
            instances = [len(train.axes[0]),len(test.axes[0])]
            count_null = int(all.isnull().sum().sum())
            if dataset in regres:
              number_class = None
              balance = None
              pos = None
              task = 'regression'
              labels_d = None
            else:
                number_class = len(all[target].unique())
                number_clases_train = train[target].value_counts()
                labels_d = all[target].dropna().unique().tolist()
                if number_class == 2:
                    task = 'binary'
                    if dataset =='sentiment-lexicons-es':
                      pos = 'positive'
                    elif dataset == 'twitter-human-bots':
                      pos = 'bot'
                    else:
                      pos = 1  
                else:
                    task = 'multiclass'
                    pos = None
                     
                min = number_clases_train.min()
                max = number_clases_train.max()
                balance = min/max 
        dict_[dataset] = {'n_columns':columns,
                          'n_instances':instances,
                          'null_values':count_null,
                          'targets':target,
                          'labels': labels_d,
                          'pos_label': pos,
                          'classes': number_class,
                          'task': task,
                          'balance':balance}
    
    with open('/media/amanda/DATA1/School/Thesis/implementation/benchmark actual/automl_benchmark/source/benchmark/properties.json', 'w') as fp:
        json.dump(dict_, fp,indent= 4,ensure_ascii= False)    

#create_prperties()