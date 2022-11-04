
  
from numpy import average
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score,recall_score,precision_score,balanced_accuracy_score,mean_squared_error,mean_absolute_error
from sklearn.metrics  import make_scorer
from yaml import dump
from benchmark import dataset
from benchmark import utils
from benchmark import functions_load
import os
import importlib   
import pandas as pd
import json
import io
import jsonlines

class AutoMLBench():
    
    scoring_mak = { 'accuracy' : make_scorer(accuracy_score), 
                'balanced_accuracy' : make_scorer(balanced_accuracy_score),
                'precision' : make_scorer(precision_score),
                'precision_macro' : make_scorer(precision_score, average = 'macro'),
                'precision_w' : make_scorer(precision_score, average = 'weighted'),
                
                'recall' : make_scorer(recall_score), 
                'recall_macro' : make_scorer(recall_score, average = 'macro'), 
                'recall_w' : make_scorer(recall_score, average ='weighted'),
                
                'f1_score' : make_scorer(f1_score),
                'f1_score_macro' : make_scorer(f1_score, average ='macro'),
                'f1_score_w' : make_scorer(f1_score, average ='weighted'),
                
                'roc_auc' : make_scorer(roc_auc_score),
                
                'MAE' : make_scorer(mean_absolute_error),
                'MSE' : make_scorer(mean_squared_error),
                'RMSE' : make_scorer(mean_squared_error,squared=False),
                }
    
    scoring = { 'accuracy' : (accuracy_score , None), 
                'balanced_accuracy' : (balanced_accuracy_score,None),
                'precision' : (precision_score,None),
                'precision_macro' : (precision_score,'macro'),
                'precision_w' : (precision_score, 'weighted'),
                
                'recall' : (recall_score , None), 
                'recall_macro' :(recall_score,'macro'), 
                'recall_w' : (recall_score, 'weighted'),
                
                'f1_score' : (f1_score , None) ,
                'f1_score_macro' : (f1_score,'macro'),
                'f1_score_w' : (f1_score, 'weighted'),
                
                'roc_auc' : (roc_auc_score,None),
                
                'MAE' : (mean_absolute_error,None),
                'MSE' : (mean_squared_error,None),
                'RMSE' : (mean_squared_error,False),
                }
 
    
    @classmethod
    def create_datasets(cls):
        local_path = os.path.dirname(os.path.realpath(__file__))
        info_path = os.path.join(local_path,'info.jsonl')
        _, df = cls.__change_names(local_path, reset= True)
        
        infos = cls.__write_info(local_path,info_path,reset=True)
        names = df['name'].to_list()
        urls = df['url'].to_list()
        funcs = df['func'].to_list()
        for i in range(len(names)):
            try:
                name = names[i]
                info = infos[i][name]
                inst = dataset.Dataset(name,urls[i],info,cls.return_func('.functions_load',funcs[i]))  
                utils.save_dataset_definition(inst)
            except Exception as error:
                print(error)
            
    @classmethod
    def init(cls,ret_info = False):
        local_path = os.path.dirname(os.path.realpath(__file__))
        info_path = os.path.join(local_path,'info.jsonl')
        names ,urls, funcs = cls.__read_info_bench(local_path)
        infos = None
        if ret_info :
            infos  =jsonlines.Reader(info_path)
            with open(info_path, "r", encoding="utf-8") as file:
                infos = [line for line in jsonlines.Reader(file)]
            
        return names,urls,funcs,infos
        
    @classmethod
    def __write_info(cls,local_path,info_path,reset = False):
        labels  = [['label'], ['label'], ['label'], ['label'], ['label'],['label'],['stroke'],['Class Name'],['fraudulent'],['price'] ,['final_status'] ,['Price'], ['Label'],['salary'],['score'],['score'],['is_humor','average'],['label'],['label'],['label'],['label'],['label'],['label'],['labels'],['account_type'],['question_asker_intent_understanding','question_body_critical','question_conversational','question_expect_short_answer','question_fact_seeking','question_has_commonly_accepted_answer','question_interestingness_others','question_interestingness_self','question_multi_intent','question_not_really_a_question','question_opinion_seeking','question_type_choice','question_type_compare','question_type_consequence','question_type_definition','question_type_entity','question_type_instructions','question_type_procedure','question_type_reason_explanation','question_type_spelling','question_well_written','answer_helpful', 'answer_level_of_information','answer_plausible','answer_relevance','answer_satisfaction','answer_type_instructions','answer_type_procedure','answer_type_reason_explanation', 'answer_well_written']]
        dict_ = []
        properties_path = os.path.join(local_path,'properties.json')
        columns_type_path = os.path.join(local_path,'columns_types.json')
        names,_,_,_= cls.init(False)
        if reset == True:
            with open(properties_path, 'r') as fp:
                properties = json.load(fp)
                with open(columns_type_path, 'r') as ff:
                    columns_type = json.load(ff)
                    for name,i in zip(names,range(len(names))):
                        if name == "wikiann-es" or name =='meddocan' or name =='wikineural-es' or name =='wikineural-en': 
                            dict_.append({ name: {'n_instances': [0,0], 
                                    'n_columns': 2, 
                                    'columns_type': columns_type[name],
                                    'targets': None ,'null_values': 0,
                                    'task': 'entity' ,
                                    'pos_label': None,
                                    'classes': None, 
                                    'class balance':None}})
                                
                        else: 
                            clases = properties[name]['classes']
                            if clases == 2:
                                task = 'binary'
                                if name =='sentiment-lexicons-es':
                                    pos = 'positive'
                                elif name == 'twitter-human-bots':
                                    pos = 'bot'
                                else:
                                    pos = 1    
                            elif clases == None:
                                task ='regression'
                                pos = None
                            else:
                                task = 'multiclass'
                                pos = None           
                            dict_.append({ name: {'n_instances': properties[name]['n_instances'],
                                    'n_columns': properties[name]['n_columns'],
                                    'columns_type': columns_type[name],
                                    'targets': labels[i] ,
                                    'null_values': properties[name]['null_values'], 
                                    'task': task,
                                    'pos_label': pos,
                                    'classes': clases , 
                                    'class balance': properties[name]['balance']} })
                         
            with jsonlines.open(info_path, 'w') as fp:
                fp.write_all(dict_)
            return dict_    
    
    @classmethod
    def __update_info(cls,info_path,info = None,remove= False, index= None):
        
        with open(info_path, "r", encoding="utf-8") as file:
            infos = [line for line in jsonlines.Reader(file)]
        
        if  index!= None:
            if not remove:
                infos[index] = info    
            if remove:
                print('kkdkdd')
                infos.pop(index) 
        elif not remove: 
            infos.append(info) 
        with jsonlines.open(info_path, 'w') as fp:
            fp.write_all(infos)
          
    @classmethod
    def __read_info_bench(cls,local_path):
        names_path = os.path.join(local_path,'bechmark_info.tsv')
        df = pd.read_table(names_path)
        return list(df['name']),list(df['url']),list(df['func'])
   
    @classmethod
    def __change_names(cls,local_path, dataset = None, insert = False, reset =False):
         
        archive_path = os.path.join(local_path,'bechmark_info.tsv')
        #archive_path = os.path.join(local_path,'bechmark_info.jsonl')
        
        datasets = ["paws-x-en","paws-x-es","wnli-es","wikiann-es","wikicat-es","sst-en",
                    "stroke-prediction","women-clothing","fraudulent-jobs","spanish-wine",
                    "project-kickstarter","price-book","inferes","predict-salary","stsb-en",
                    "stsb-es","haha", "meddocan","vaccine-es","vaccine-en","sentiment-lexicons-es",
                    "wikineural-en","wikineural-es","language-identification","twitter-human-bots","google-guest"]
        urls = ["https://github.com/amyGB99/automl_benchmark/releases/download/paws-x/paws-x-en.zip",
            "https://github.com/amyGB99/automl_benchmark/releases/download/paws-x/paws-x-es.zip",
            "https://github.com/amyGB99/automl_benchmark/releases/download/wnli-es/wnli-es.zip",
            "https://github.com/amyGB99/automl_benchmark/releases/download/wikiann-es/wikiann-es.zip",
            "https://github.com/amyGB99/automl_benchmark/releases/download/wikicat-es/wikicat-es.zip",
            "https://github.com/amyGB99/automl_benchmark/releases/download/sst-en/sst-en.zip",
            "https://github.com/amyGB99/automl_benchmark/releases/download/stroke-prediction/stroke-prediction.zip",
            "https://github.com/amyGB99/automl_benchmark/releases/download/women-clothing/women-clothing.zip",
            "https://github.com/amyGB99/automl_benchmark/releases/download/fraudulent-jobs/fraudulent-jobs.zip",
            "https://github.com/amyGB99/automl_benchmark/releases/download/spanish-wine/spanish-wine.zip",
            "https://github.com/amyGB99/automl_benchmark/releases/download/project-kickstarter/project-kickstarter.zip",
            "https://github.com/amyGB99/automl_benchmark/releases/download/price-book/price-book.zip",
            "https://github.com/amyGB99/automl_benchmark/releases/download/inferes/inferes.zip",
            "https://github.com/amyGB99/automl_benchmark/releases/download/predict-salary/predict-salary.zip",
            "https://github.com/amyGB99/automl_benchmark/releases/download/stsb/stsb-en.zip",
            "https://github.com/amyGB99/automl_benchmark/releases/download/stsb/stsb-es.zip",
            "https://github.com/amyGB99/automl_benchmark/releases/download/haha/haha.zip",
            "https://github.com/amyGB99/automl_benchmark/releases/download/meddocan/meddocan.zip",
            "https://github.com/amyGB99/automl_benchmark/releases/download/vaccine/vaccine-es.zip",
            "https://github.com/amyGB99/automl_benchmark/releases/download/vaccine/vaccine-en.zip",
            "https://github.com/amyGB99/automl_benchmark/releases/download/sentiment-lexicons/sentiment-lexicons-es.zip",
            "wikineural-en",
            "wikineural-es",
            "https://github.com/amyGB99/automl_benchmark/releases/download/language-identification/language-identification.zip",
            "https://github.com/amyGB99/automl_benchmark/releases/download/twitter-human-bots/twitter-human-bots.zip",
            "https://github.com/amyGB99/automl_benchmark/releases/download/google-guest/google-guest.zip"]
    
        func = ["load_paws","load_paws","load_wnli","load_wikiann","load_wikicat",
                 "load_sst_en","load_stroke", "load_women_clothing" , "load_jobs","load_wines","load_project_kickstarter",
                 "load_price_book","load_inferes", "load_predict_salary","load_stsb","load_stsb","load_haha","load_meddocan",
                 "load_vaccine","load_vaccine","load_sentiment","load_wikineural","load_wikineural","load_language",
                 "load_twitter_human","load_google_guest"]
        if reset:
            df = pd.DataFrame()
            df['name'] = datasets
            df['url'] = urls
            df['func'] = func
        else:
            df = pd.read_table(archive_path)
               
        names = df['name'].tolist()
        try:
            index = names.index(dataset['name'])   
        except:
            index = None
        if dataset is not None:
            if index != None:
                if insert:
                    df.iloc[index,:] = dataset
                else:
                    df = df.drop([index],axis=0).reset_index(drop=True)         
            else:    
                if insert:
                    df = df.append(dataset,ignore_index = True)
        elif not reset:
            index = None
            print(f"Error the dataset")
        df.to_csv(archive_path,sep='\t',index=False) 
        return index,df 
    
    @classmethod
    def return_func(cls,mod, func):
        module = importlib.import_module(mod,package="benchmark")
        function = getattr(module, func)
        return function
    
    @classmethod
    def caller_func(cls,mod, func:str, *args):
        function = cls.return_func(mod,func)
        function(*args)
    
    @classmethod
    def load_dataset(cls, name, format = "pandas", in_xy = True, samples = 2,encoding = 'utf-8'):
        dataset = utils.load_dataset_definition(name)
        if dataset is not None:
            try:
                return dataset.loader_func(dataset,format , in_xy , samples, encoding)
            except Exception as error :
               print(error)
        return None
    
    @classmethod
    def load_info(cls,name):
        try:
            return utils.load_dataset_definition(name).info
        except:
            print("There is no registered dataset with that name")
            return None
    
    @classmethod
    def filter(cls, task = None,expresion = None):
        '''
        task: str = 'binary', 'regression,'multiclass'
        expression: tuple(len(3)) = (property,min,max) : min <= property < max
        
        return : list[str]
        '''  
        names,_,_,infos = cls.init(True)
        if task == None and expresion == None:
            return names
        else:
            list_ = []
            for info,name in zip(infos,names):
                value = info[name]
                if value is None:
                    continue
                try:
                    if task != None and expresion == None:
                        if task == value['task']:
                            list_.append(name) 
                        continue
                    elif expresion != None:
                        if task != None and task != value['task']:
                            continue
                        property,min,max = expresion
                        value_property = value[property]
                        if min == None and max == None:
                            print('Both ranges cannot be none')
                            break
                        try:

                            if value_property ==None :
                                continue
                            if min != None and max != None:
                                if value_property >= min and value_property < max:
                                    list_.append(name)
                            elif min == None:
                                if value_property < max:
                                    list_.append(name)
                            elif max == None:
                                if value_property >= min:
                                    list_.append(name)           
                        except:
                            print('The feature is incorrect')
                            break            
                except Exception as error:
                    print(error)
                    break
            return list_        
                                         
    @classmethod
    def new_dataset(cls,name: str, url: str, function: str , info = None ):
        '''
        info = {'n_instances': [int,int],
                'n_columns': int , 
                'columns_type': {'name' : type},
                'targets': list[str] ,
                'null_values': int,
                'task': str 
                'pos_label': Any,
                'classes': int, 
                'class balance':float}
        '''
        correct = False
        #names,_,_,_= cls.init(True) 

        try:
            inst = dataset.Dataset(name,url,info,cls.return_func('.functions_load',function))
            correct = True
        except:
            try:
                inst = dataset.Dataset(name,url,info,function) 
                correct = True
            except Exception as error:
                print(error) 
        if correct:    
            local_path = os.path.dirname(os.path.realpath(__file__))
            info_path = os.path.join(local_path,'info.jsonl')
            try:
                new_fila = {'name':name, 'url': url,'func': inst.loader_func_name}
            except:    
                new_fila =  {'name': name, 'url': url,'func': None}
            if info is None:
                info = { name:{'n_instances': None, 
                    'n_columns': None, 
                    'columns_type': None,
                    'targets': None ,'null_values': None,
                    'task': None,
                    'pos_label': None,
                    'classes': None, 
                    'class balance':None}
                }
            else: 
                info = { name: info}   
            try:          
                index,_ = cls.__change_names(local_path, dataset = new_fila,insert = True)
                cls.__update_info(info_path, info = info,remove = False, index = index)   
                utils.save_dataset_definition(inst)
            except Exception as error:
                print(error)
                
                    
    @classmethod
    def remove_dataset(cls, name: str):
        #names,_,_,infos= cls.init(True) 
        
        local_path = os.path.dirname(os.path.realpath(__file__))
        info_path = os.path.join(local_path,'info.jsonl')
        row  ={'name': name}
        try:
            index,_ = cls.__change_names(local_path, dataset= row,insert= False,reset =False)
            print(index)
            cls.__update_info(info_path,remove= True,index= index)
        except:
            print('kakaka')
            cls.__change_names(local_path,reset = True)
            cls.__write_info(local_path,info_path,reset=True)

# class Metric():
#     @classmethod    
#     def auroc(cls,**kwargs):
#         auroc = make_scorer(name= 'roc_auc',
#         score_func=roc_auc_score,
#         optimum= 1,
#         greater_is_better=True,**kwargs)
#         return auroc
#     @classmethod
#     def f1(cls):
#         f1 = make_scorer(f1_score, average='weighted')
#         return f1
#     @classmethod
#     def accuracy(cls,**kwargs):
#         accuracy = make_scorer(name= 'accuracy',
#         score_func= accuracy_score,
#         optimum= 1,
#         greater_is_better=True,**kwargs) 
#         return accuracy   
 
        
            
        