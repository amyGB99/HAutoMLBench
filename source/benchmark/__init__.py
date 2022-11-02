
  
from numpy import average
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics  import make_scorer
from benchmark import dataset
from benchmark import utils
from benchmark import functions_load
import os
import importlib   
import pandas as pd
import json

class AutoMLBench():
    
    names = []
    inst = {}
    info = {}
    
    @classmethod
    def init(cls):
        local_path = os.path.dirname(os.path.realpath(__file__))
        info_path = os.path.join(local_path,'info.json')
        cls._write_info(local_path,info_path)
        with open(info_path, 'r') as fp:
            cls.info = json.load(fp)
        cls.names = list(cls.info.keys())
        for name in cls.names:
           inst = dataset.Dataset(name,cls.info[name]['url'],cls.info[name],cls.return_func('.functions_load',cls.info[name]['func']))  
           cls.inst[name] = inst
           utils.save_dataset_definition(inst)
    
    @classmethod
    def _write_info(cls,local_path,info_path):
        properties_path = os.path.join(local_path,'properties.json')
        columns_type_path = os.path.join(local_path,'columns_types.json')
        labels  = [['label'], ['label'], ['label'], ['label'], ['label'],['label'],['stroke'],['Class Name'],['fraudulent'],['price'] ,['final_status'] ,['Price'], ['Label'],['salary'],['score'],['score'],['is_humor','average'],['label'],['label'],['label'],['label'],['label'],['label'],['labels'],['account_type'],['question_asker_intent_understanding','question_body_critical','question_conversational','question_expect_short_answer','question_fact_seeking','question_has_commonly_accepted_answer','question_interestingness_others','question_interestingness_self','question_multi_intent','question_not_really_a_question','question_opinion_seeking','question_type_choice','question_type_compare','question_type_consequence','question_type_definition','question_type_entity','question_type_instructions','question_type_procedure','question_type_reason_explanation','question_type_spelling','question_well_written','answer_helpful', 'answer_level_of_information','answer_plausible','answer_relevance','answer_satisfaction','answer_type_instructions','answer_type_procedure','answer_type_reason_explanation', 'answer_well_written']]
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
        with open(properties_path, 'r') as fp:
            properties = json.load(fp)
            with open(columns_type_path, 'r') as ff:
                columns_type = json.load(ff)
                dict_ = {}
                for name,i in zip (datasets, range(len(labels))):
                    if name == "wikiann-es" or name =='meddocan' or name =='wikineural-es' or name =='wikineural-en': 
                        dict_[name]= { 'url': urls[i],  
                                        'func': func[i],
                                        'columns_type': columns_type[name],
                                        'properties':{'n_columns': 2 , 'n_instances': [0,0],'targets': None ,'null_values': 0,'task': 'entity' ,'classes': None, 'class balance':None}
                                    }
                    else: 
                        clases = properties[name]['classes']
                        if clases == 2:
                            task = 'binary'
                        elif clases == None:
                            task ='regression'
                        else:
                            task = 'multiclass'            
                        dict_[name]= { 'url': urls[i],  
                                    'func': func[i],
                                    'columns_type': columns_type[name],
                                    'properties':{'n_columns': properties[name]['n_columns'], 'n_instances': properties[name]['n_instances'],'targets': labels[i] ,'null_values': properties[name]['null_values'], 'task': task,'classes': clases , 'class balance': properties[name]['balance']}
                                    }   
                    
            
        with open(info_path, 'w') as fp:
            json.dump(dict_, fp,indent= 4)   
    
    @classmethod
    def return_func(cls,mod, func,package ="benchmark"):
        module = importlib.import_module(mod,package)
        function = getattr(module, func)
        return function
    
    @classmethod
    def caller_func(cls,mod, func, *args):
        function = cls.return_func(mod,func)
        function(*args)
    
    @classmethod
    def load_dataset(cls, name, format = "pandas", in_xy = True, samples = 2,encoding = 'utf-8'):
        dataset = utils.load_dataset_definition(name)
        if dataset is not None:
            try:
                return dataset.loader_func(cls.inst[name],format , in_xy , samples, encoding)
            except Exception as error :
               print(error)
        return None
    
    @classmethod
    def load_info(cls,name):
        try:
            return cls.info[name]
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
        
        if task == None and expresion == None:
            return cls.names
        elif expresion == None:
            return [ key  for (key,value) in cls.info.items() if value['properties'] is not None and value['properties']['task']== task]
        else:
            list_ = []
            for (key,dicts) in cls.info.items():
                if dicts['properties'] is None:
                    continue
                value = dicts['properties']
                try:
                    property,min,max = expresion
                    if task != None:
                        if task != value['task']:
                            continue 
                    if min == None and max == None:
                        print('Both ranges cannot be none')
                        break
                    try:
                        value1 = value[property]
                        if value1 ==None :
                            continue
                        if min != None and max != None:
                            if value1 >= min and value1 < max:
                                list_.append(key)
                        elif min == None:
                            if value1 < max:
                                list_.append(key)
                        elif max == None:
                            if value1 >= min:
                                list_.append(key)           
                    except:
                        print('The feature is incorrect')
                        break            
                except Exception as error:
                    print(error)
                    break
            return list_        
                                         
    @classmethod
    def new_dataset(cls,name: str, url: str,function ,permanent=  False, info = None ):
        '''
        info = {'columns_type': {'name' : type},
                'properties':{'n_columns': int , 'n_instances': [int,int],'targets': list[str] ,'null_values': int,'task': str ,'classes': int, 'class balance':float}}
        '''
        if name not in cls.names:
            cls.names.append(name)
            if info is not None:
                cls.info[name] = { 'url': url,  
                                    'func': function.__name__,
                                    'columns_type': info['columns_type'],
                                    'properties':info['properties']
                                }
            else:
                cls.info[name] = { 'url': url,  
                                    'func': function.__name__,
                                    'columns_type':None,
                                    'properties':None
                                }
            if permanent:
                local_path = os.path.dirname(os.path.realpath(__file__))
                info_path = os.path.join(local_path,'info.json')
                with open(info_path, 'w') as fp:
                    json.dump(cls.info, fp,indent= 4)   
            inst = dataset.Dataset(name,url,cls.info[name],function)
            cls.inst[name] = inst
            try:
                utils.save_dataset_definition(inst)
            except Exception as error:
                print(error)
        else:
            print('A dataset with that name already exists')        
                    
    @classmethod
    def remove_dataset(cls, name: str,permanent=  False):
        if name in cls.names:
            cls.names.remove(name)
            cls.inst.pop(name)
            cls.info.pop(name)
            if permanent:
                local_path = os.path.dirname(os.path.realpath(__file__))
                info_path = os.path.join(local_path,'info.json')
                with open(info_path, 'w') as fp:
                    json.dump(cls.info, fp,indent= 4)      
class Metric():
    @classmethod    
    def auroc(cls,**kwargs):
        auroc = make_scorer(name= 'roc_auc',
        score_func=roc_auc_score,
        optimum= 1,
        greater_is_better=True,**kwargs)
        return auroc
    @classmethod
    def f1(cls,**kwargs):
        f1 = make_scorer(name= 'f1',
        score_func= f1_score,
        optimum= 1,
        greater_is_better=True,average='weighted',**kwargs)
        return f1
    @classmethod
    def accuracy(cls,**kwargs):
        accuracy = make_scorer(name= 'accuracy',
        score_func= accuracy_score,
        optimum= 1,
        greater_is_better=True,**kwargs) 
        return accuracy   
 
        
            
        