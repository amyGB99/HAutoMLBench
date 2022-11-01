from cmath import inf
from distutils.log import info
from black import err
from numpy import average
    
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics  import make_scorer
from benchmark import dataset
from benchmark import utils
from benchmark import functions_load
import os
import importlib   
#from .utils import save_dataset_definition,load_dataset_definition
import pandas as pd
import json

class AutoMLBench():
    
    names = []
    name_func = []
    intances = {}
    inst = {}
    info = []
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
    
    columns = {"paws-x-en": {'sentence1':'text','sentence2':'text', 'label':'categorical'},
                    "paws-x-es":{'sentence1':'text','sentence2':'text', 'label':'categorical'},
                    "wnli-es":{'sentence1':'text','sentence2':'text', 'label':'categorical'},
                    "wikiann-es":{'tokens':'Seqtokens','tags':'Seqtags'}, 
                    "wikicat-es":['text','categorical'],
                    "sst-en": ['categorical','text'],
                    "stroke-prediction": ['categorical','int','categorical','categorical','categorical','categorical','categorical','float','float','categorical','categorical'],
                    "women-clothing": ['int','text','text','category','category','category','category','category','category'],
                    "fraudulent-jobs":['text','string','string','string','text','text','text','text','category','category','category','string','string','string','string','category'],
                    "spanish-wine":['string','string','int','float','int','string','string','float','category','category','category'], 
                    "project-kickstarter":[],
                    "price-book":[],
                    "inferes":['sentence','sentence','category','int','int','category'],
                    "predict-salary":[],
                    "stsb-en":['sentence','sentence','float'], 
                    "stsb-es":['sentence','sentence','float'],
                    "haha":['sentence','category'], 
                    "meddocan":{'tokens':'Seqtokens','tags':'Seqtags'},
                    "vaccine-es":['text','categorical'],
                    "vaccine-en":['text','categorical'],
                    "sentiment-lexicons-es":['text','categorical'],
                    "wikineural-en":{'tokens':'Seqtokens','tags':'Seqtags'},
                    "wikineural-es":{'tokens':'Seqtokens','tags':'Seqtags'},
                    "language-identification":['text','categorical'],
                    'twitter-human-bots': { 'created_at': 'datetime' ,'default_profile': 'boolean' ,'default_profile_image': 'boolean','description': 'text','favourites_count':'int','followers_count':'int','friends_count':'int','geo_enabled':'boolean','lang':'string','location':'string','profile_background_image_url':'image_url','profile_image_url':'image_url','screen_name':'text','statuses_count':'int','verified':'boolean' ,'average_tweets_per_day':'float' ,'account_age_days':'int' ,'account_type':'category'}, 
                    'google-guest': ['text','text','text','text','text','text','text','text','categorical','text','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float']}
    
    # info = { "paws-x-en":{ 'n_columns': 3, 'n_instances': [49401,4000],'targets': ['label'] ,'null_values': False, 'classes': 2, 'class imbalance': 0.23},
    #         "paws-x-es":{ 'n_columns': 3 ,'n_instances': [49401,4000],'targets': ['label'] ,'null_values': True, 'classes': 2, 'class imbalance': 0.23},
    #         "wnli-es":{ 'n_columns': 3, 'n_instances': [635,70],'targets': ['label'], 'null_values': False,'classes': 2, 'class imbalance': 0.23 },
    #         "wikiann-es":{ 'n_columns': 0,'n_instances': [0,0] , 'targets': ['label'], 'null_values': False,'classes': None, 'class imbalance': 0.23 },
    #         "wikicat-es":{ 'n_columns': 2,'n_instances': [7909,3402] , 'targets': ['label'], 'null_values': False,'classes': 2, 'class imbalance': 0.23 },
    #         "sst-en":{ 'n_columns': 2,'n_instances': [8544,2210] , 'targets': ['label'], 'null_values': False,'classes': 2, 'class imbalance': 0.23 },
    #         "stroke-prediction": { 'n_columns': 11,'n_instances': [4088,1022] , 'targets': ['stroke'], 'null_values': True,'classes': 2, 'class imbalance': 0.23 },
    #         "women-clothing": { 'n_columns': 9,'n_instances': [16440,7046] , 'targets': ['Class Name'], 'null_values': True,'classes': 2, 'class imbalance': 0.23 },
    #         "fraudulent-jobs": { 'n_columns': 17,'n_instances': [12516,5304] , 'targets': ['fraudulent'], 'null_values': True,'classes': 2, 'class imbalance': 0.23 },
    #         "spanish-wine": { 'n_columns': 11,'n_instances': [6000,1612] , 'targets': ['price'], 'null_values': True,'classes': None, 'class imbalance': 0.23 },
    #         "project-kickstarter": { 'n_columns': 12,'n_instances': [108129,63465] , 'targets': ['final_status'], 'null_values': True,'classes': 2, 'class imbalance': 0.23 },
    #         "price-book": { 'n_columns': 9,'n_instances': [6237,1650] , 'targets': ['Price'], 'null_values': True,'classes': None, 'class imbalance': 0.23 },
    #         "inferes": { 'n_columns': 6,'n_instances': [6444,1612] , 'targets': ['Label'], 'null_values': True,'classes': 2, 'class imbalance': 0.23 },
    #         "predict-salary": { 'n_columns': 8,'n_instances': [19802,6001] , 'targets': ['salary'], 'null_values': True,'classes': 2, 'class imbalance': 0.23 },
    #         "stsb-en": { 'n_columns': 3,'n_instances': [7249,1378] , 'targets': ['score'], 'null_values': False,'classes': None, 'class imbalance': None },
    #         "stsb-es": { 'n_columns': 3,'n_instances': [7249,1378] , 'targets': ['score'], 'null_values': False,'classes': None, 'class imbalance': None},
    #         "haha": { 'n_columns': 2,'n_instances': [24000,6000] , 'targets': ['is_humor','average'], 'null_values': True,'classes': 2, 'class imbalance': 0.23 },
    #         "meddocan": { 'n_columns': 0,'n_instances': [0,0] , 'targets': ['label'], 'null_values': True,'classes': 2, 'class imbalance': 0.23 },
    #         "vaccine-es": { 'n_columns': 2,'n_instances': [2003,694] , 'targets': ['label'], 'null_values': True,'classes': 2, 'class imbalance': 0.23 },
    #         "vaccine-en": { 'n_columns': 2,'n_instances': [1770,312] , 'targets': ['label'], 'null_values': True,'classes': 2, 'class imbalance': 0.23 },
    #         "sentiment-lexicons-es": { 'n_columns': 2,'n_instances': [4075,200] , 'targets': ['label'], 'null_values': False,'classes': 2, 'class imbalance': 0.23 },
    #         "wikineural-en": { 'n_columns': 0,'n_instances': [0,0] , 'targets': ['label'], 'null_values': True,'classes': 2, 'class imbalance': 0.23 },
    #         "wikineural-es": { 'n_columns': 0,'n_instances': [0,0] , 'targets': ['label'], 'null_values': True,'classes': 2, 'class imbalance': 0.23 },
    #         "language-identification": { 'n_columns': 2,'n_instances': [80000,10000] , 'targets': ['labels'], 'null_values': False,'classes': 20, 'class imbalance': 0.23 },
    #         "twitter-human-bots": { 'n_columns': 18,'n_instances': [29950,7488] , 'targets': ['account_type'], 'null_values': True,'classes': 2, 'class imbalance': 0.23 },
            
    #        "google-guest": { 'n_columns': 40,'n_instances': [6079,476] ,'targets':['question_asker_intent_understanding','question_body_critical','question_conversational','question_expect_short_answer','question_fact_seeking','question_has_commonly_accepted_answer','question_interestingness_others','question_interestingness_self','question_multi_intent','question_not_really_a_question','question_opinion_seeking','question_type_choice',
    #         'question_type_compare','question_type_consequence','question_type_definition','question_type_entity','question_type_instructions','question_type_procedure','question_type_reason_explanation','question_type_spelling','question_well_written','answer_helpful', 'answer_level_of_information','answer_plausible','answer_relevance','answer_satisfaction','answer_type_instructions','answer_type_procedure','answer_type_reason_explanation', 'answer_well_written'], 'null_values': False,'classes': None, 'class imbalance': None },
            
            
            
            
    #         }
           
    
    func = ["load_paws","load_paws","load_wnli","load_wikiann","load_wikicat",
                 "load_sst_en","load_stroke", "load_women_clothing" , "load_jobs","load_wines","load_project_kickstarter",
                 "load_price_book","load_inferes", "load_predict_salary","load_stsb","load_stsb","load_haha","load_meddocan",
                 "load_vaccine","load_vaccine","load_sentiment","load_wikineural","load_wikineural","load_language",
                 "load_twitter_human","load_google_guest"]
    
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
        features_path = os.path.join(local_path,'columns_types.json')
        a  = [['label'], ['label'], ['label'], ['label'], ['label'],['label'],['stroke'],['Class Name'],['fraudulent'],['price'] ,['final_status'] ,['Price'], ['Label'],['salary'],['score'],['score'],['is_humor','average'],['label'],['label'],['label'],['label'],['label'],['label'],['labels'],['account_type'],['question_asker_intent_understanding','question_body_critical','question_conversational','question_expect_short_answer','question_fact_seeking','question_has_commonly_accepted_answer','question_interestingness_others','question_interestingness_self','question_multi_intent','question_not_really_a_question','question_opinion_seeking','question_type_choice','question_type_compare','question_type_consequence','question_type_definition','question_type_entity','question_type_instructions','question_type_procedure','question_type_reason_explanation','question_type_spelling','question_well_written','answer_helpful', 'answer_level_of_information','answer_plausible','answer_relevance','answer_satisfaction','answer_type_instructions','answer_type_procedure','answer_type_reason_explanation', 'answer_well_written']]
        with open(properties_path, 'r') as fp:
            properties = json.load(fp)
            with open(features_path, 'r') as ff:
                features = json.load(ff)
                dict_ = {}
                for name,i in zip (cls.datasets, range(len(a))):
                    if name == "wikiann-es" or name =='meddocan' or name =='wikineural-es' or name =='wikineural-en': 
                        dict_[name]= { 'url': cls.urls[i],  
                                        'func': cls.func[i],
                                        'features': features[name],
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
                        dict_[name]= { 'url': cls.urls[i],  
                                    'func': cls.func[i],
                                    'features': features[name],
                                    'properties':{'n_columns': properties[name]['n_columns'], 'n_instances': properties[name]['n_instances'],'targets': a[i] ,'null_values': properties[name]['null_values'], 'task': task,'classes': clases , 'class balance': properties[name]['balance']}
                                        }   
                    
            
        with open(info_path, 'w') as fp:
            json.dump(dict_, fp,indent= 4)   
    
    @classmethod
    def return_func(cls,mod, func):
        module = importlib.import_module(mod,package="benchmark")
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
            return [ key  for (key,value) in cls.info.items() if value['task']== task]
        else:
            list_ = []
            for (key,dicts) in cls.info.items():
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
    def new_dataset(cls,name: str, url: str,info = None,function= None):
        # try:
        cls.names.append(name)
        cls.urls.append(url)
        inst = dataset.Dataset(name,url,info,function)
        cls.inst[f'{name}'] = inst
        cls.info.append(info)
        utils.save_dataset_definition(inst)
        cls.name_func.append(function.__name__)
        # except:
        #     print(f"Error in introduce a {name} dataset.")           
#       
class Metric():
    @classmethod    
    def auroc(cls):
        auroc = make_scorer(name= 'roc_auc',
        score_func=roc_auc_score,
        optimum= 1,
        greater_is_better=True)
        return auroc
    @classmethod
    def f1(cls):
        f1 = make_scorer(name= 'f1',
        score_func= f1_score,
        optimum= 1,
        greater_is_better=True)
        return f1
    @classmethod
    def accuracy(cls):
        accuracy = make_scorer(name= 'accuracy',
        score_func= accuracy_score,
        optimum= 1,
        greater_is_better=True) 
        return accuracy   
 
        
            
        