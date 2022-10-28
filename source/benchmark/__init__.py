from cmath import inf
from distutils.log import info
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
    urls = []
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
    
    columns = {"paws-x-en": {'sentence1':'text','sentence1':'text', 'label':'categorical'},
                    "paws-x-es":{'sentence1':'text','sentence1':'text', 'label':'categorical'},
                    "wnli-es":{'sentence1':'text','sentence1':'text', 'label':'categorical'},
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
                    "meddocan":['SeqTokens','SeqTags'],
                    "vaccine-es":['text','categorical'],
                    "vaccine-en":['text','categorical'],
                    "sentiment-lexicons-es":['text','categorical'],
                    "wikineural-en":{'tokens':'Seqtokens','tags':'Seqtags'},
                    "wikineural-es":{'tokens':'Seqtokens','tags':'Seqtags'},
                    "language-identification":['text','categorical'],
                    'twitter-human-bots': { 'created_at': 'datetime' ,'default_profile': 'boolean' ,'default_profile_image': 'boolean','description': 'text','favourites_count':'int','followers_count':'int','friends_count':'int','geo_enabled':'boolean','lang':'string','location':'string','profile_background_image_url':'image_url','profile_image_url':'image_url','screen_name':'text','statuses_count':'int','verified':'boolean' ,'average_tweets_per_day':'float' ,'account_age_days':'int' ,'account_type':'category'}, 
                    'google-guest': ['text','text','text','text','text','text','text','text','categorical','text','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float']}
    
    
    info = { "paws-x-en":{ 'n_columns': 3, 'n_instances': [49401,4000],'targets': ['label'] ,'null_values': False, 'classes': 2, 'class imbalance': 0.23},
            "paws-x-es":{ 'n_columns': 3 ,'n_instances': [49401,4000],'targets': ['label'] ,'null_values': True, 'classes': 2, 'class imbalance': 0.23},
            "wnli-es":{ 'n_columns': 3, 'n_instances': [635,70],'targets': ['label'], 'null_values': False,'classes': 2, 'class imbalance': 0.23 },
            "wikiann-es":{ 'n_columns': 0,'n_instances': [0,0] , 'targets': ['label'], 'null_values': False,'classes': None, 'class imbalance': 0.23 },
            "wikicat-es":{ 'n_columns': 3,'n_instances': [7908,3402] , 'targets': ['label'], 'null_values': False,'classes': 2, 'class imbalance': 0.23 },
            "sst-en":{ 'n_columns': 2,'n_instances': [8544,2210] , 'targets': ['label'], 'null_values': False,'classes': 2, 'class imbalance': 0.23 },
            "stroke-prediction": { 'n_columns': 2,'n_instances': [4088,1022] , 'targets': ['fraudulent'], 'null_values': True,'classes': 2, 'class imbalance': 0.23 },
            "women-clothing": { 'n_columns': 2,'n_instances': [16440,7046] , 'targets': ['label'], 'null_values': True,'classes': 2, 'class imbalance': 0.23 },
            "fraudulent-jobs": { 'n_columns': 2,'n_instances': [12516,5304] , 'targets': ['label'], 'null_values': True,'classes': 2, 'class imbalance': 0.23 },
            "spanish-wine": { 'n_columns': 2,'n_instances': [6000,1612] , 'targets': ['label'], 'null_values': True,'classes': 2, 'class imbalance': 0.23 },
            "project-kickstarter": { 'n_columns': 2,'n_instances': [108129,63465] , 'targets': ['label'], 'null_values': True,'classes': 2, 'class imbalance': 0.23 },
            "price-book": { 'n_columns': 2,'n_instances': [0,0] , 'targets': ['label'], 'null_values': True,'classes': 2, 'class imbalance': 0.23 },
            "inferes": { 'n_columns': 2,'n_instances': [6444,1612] , 'targets': ['label'], 'null_values': True,'classes': 2, 'class imbalance': 0.23 },
            "predict-salary": { 'n_columns': 2,'n_instances': [19802,6001] , 'targets': ['label'], 'null_values': True,'classes': 2, 'class imbalance': 0.23 },
            "stsb-en": { 'n_columns': 3,'n_instances': [7249,1378] , 'targets': ['score'], 'null_values': False,'classes': None, 'class imbalance': None },
            "stsb-es": { 'n_columns': 3,'n_instances': [7249,1378] , 'targets': ['score'], 'null_values': False,'classes': None, 'class imbalance': None},
            "haha": { 'n_columns': 2,'n_instances': [24000,6000] , 'targets': ['is_humor','average'], 'null_values': True,'classes': 2, 'class imbalance': 0.23 },
            "meddocan": { 'n_columns': 2,'n_instances': [0,0] , 'targets': ['label'], 'null_values': True,'classes': 2, 'class imbalance': 0.23 },
            "vaccine-es": { 'n_columns': 2,'n_instances': [2003,694] , 'targets': ['label'], 'null_values': True,'classes': 2, 'class imbalance': 0.23 },
            "vaccine-en": { 'n_columns': 2,'n_instances': [1770,312] , 'targets': ['label'], 'null_values': True,'classes': 2, 'class imbalance': 0.23 },
            "sentiment-lexicons-es": { 'n_columns': 2,'n_instances': [0,0] , 'targets': ['label'], 'null_values': True,'classes': 2, 'class imbalance': 0.23 },
            "wikineural-en": { 'n_columns': 2,'n_instances': [0,0] , 'targets': ['label'], 'null_values': True,'classes': 2, 'class imbalance': 0.23 },
            "wikineural-es": { 'n_columns': 2,'n_instances': [0,0] , 'targets': ['label'], 'null_values': True,'classes': 2, 'class imbalance': 0.23 },
            "language-identification": { 'n_columns': 2,'n_instances': [80000,10000] , 'targets': ['label'], 'null_values': True,'classes': 2, 'class imbalance': 0.23 },
            "twitter-human-bots": { 'n_columns': 2,'n_instances': [29950,7488] , 'targets': ['account_type'], 'null_values': True,'classes': 2, 'class imbalance': 0.23 },
            
           "google-guest": { 'n_columns': 40,'n_instances': [6079,476] ,'targets':['question_asker_intent_understanding','question_body_critical,question_conversational','question_expect_short_answer','question_fact_seeking','question_has_commonly_accepted_answer','question_interestingness_others','question_interestingness_self','question_multi_intent','question_not_really_a_question','question_opinion_seeking','question_type_choice',
            'question_type_compare','question_type_consequence','question_type_definition','question_type_entity','question_type_instructions','question_type_procedure','question_type_reason_explanation','question_type_spelling','question_well_written','answer_helpful', 'answer_level_of_information','answer_plausible','answer_relevance','answer_satisfaction','answer_type_instructions','answer_type_procedure','answer_type_reason_explanation', 'answer_well_written'], 'null_values': False,'classes': 0, 'class imbalance': None },
            
            
            
            
            }
           
    
    func = ["load_paws","load_paws","load_wnli","load_wikiann","load_wikicat",
                 "load_sst_en","load_stroke", "load_women_clothing" , "load_jobs","load_wines","load_project_kickstarter",
                 "load_price_book","load_inferes", "load_predict_salary","load_stsb","load_stsb","load_haha","load_meddocan",
                 "load_vaccine","load_vaccine","load_sentiment","load_wikineural","load_wikineural","load_language",
                 "load_twitter_human","load_google_guest"]
    
    @classmethod
    def init(cls):
        local_path = os.path.dirname(os.path.realpath(__file__))
        archive_path = os.path.join(local_path,'list_datasets.txt')
        info_path = os.path.join(local_path,'info.json')
        #cls._write_archive(archive_path,info_path)
        with open(info_path, 'r') as fp:
            cls.info = json.load(fp)
        datasets_list = pd.read_csv(archive_path,sep="\t")
        cls.names = list(datasets_list["name"])
        cls.name_func = list(datasets_list["func"])
        cls.urls = list(datasets_list["url"])
        for i in range(len(cls.names)):
           inst = dataset.Dataset(cls.names[i],cls.urls[i],cls.info[cls.names[i]],cls.return_func('.functions_load',cls.name_func[i]))  
           cls.inst[f'{cls.names[i]}'] = inst
           utils.save_dataset_definition(inst)
    
    @classmethod
    def _write_archive(cls,list_,info_path):  
        data = pd.DataFrame()
        data["name"] = cls.datasets
        data["url"] = cls.urls
        data["func"] = cls.func
        data.to_csv(list_,sep="\t",index= False)
        with open(info_path, 'w') as fp:
            json.dump(cls.info, fp,indent= 4)   
    
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
        return dataset.loader_func(cls.inst[name],format , in_xy , samples, encoding)
    
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