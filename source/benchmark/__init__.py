from benchmark import dataset
from benchmark import utils
from benchmark import functions_load
import os
import importlib   
#from .utils import save_dataset_definition,load_dataset_definition
import pandas as pd

class AutoMLBench():
    
    names = []
    urls = []
    name_func = []
    intances = {}
    inst = {}
    # datasets = ["paws-x-en","paws-x-es","wnli-es","wikiann-es","wikicat-es","sst-en",
    #                  "stroke-prediction","women-clothing","fraudulent-jobs","spanish-wine",
    #                  "project-kickstarter","price-book","inferes","predict-salary","stsb-en",
    #                  "stsb-es","haha", "meddocan"]
    
    # urls = ["https://github.com/amyGB99/automl_benchmark/releases/download/paws-x/paws-x-en.zip",
    #             "https://github.com/amyGB99/automl_benchmark/releases/download/paws-x/paws-x-es.zip",
    #             "https://github.com/amyGB99/automl_benchmark/releases/download/wnli-es/wnli-es.zip",
    #             "https://github.com/amyGB99/automl_benchmark/releases/download/wikiann-es/wikiann-es.zip",
    #             "https://github.com/amyGB99/automl_benchmark/releases/download/wikicat-es/wikicat-es.zip",
    #             "https://github.com/amyGB99/automl_benchmark/releases/download/sst-en/sst-en.zip",
    #             "https://github.com/amyGB99/automl_benchmark/releases/download/stroke-prediction/stroke-prediction.zip",
    #             "https://github.com/amyGB99/automl_benchmark/releases/download/women-clothing/women-clothing.zip",
    #             "https://github.com/amyGB99/automl_benchmark/releases/download/fraudulent-jobs/fraudulent-jobs.zip",
    #             "https://github.com/amyGB99/automl_benchmark/releases/download/spanish-wine/spanish-wine.zip",
    #             "https://github.com/amyGB99/automl_benchmark/releases/download/project-kickstarter/project-kickstarter.zip",
    #             "https://github.com/amyGB99/automl_benchmark/releases/download/price-book/price-book.zip",
    #             "https://github.com/amyGB99/automl_benchmark/releases/download/inferes/inferes.zip",
    #             "https://github.com/amyGB99/automl_benchmark/releases/download/predict-salary/predict-salary.zip",
    #             "https://github.com/amyGB99/automl_benchmark/releases/download/stsb/stsb-en.zip",
    #             "https://github.com/amyGB99/automl_benchmark/releases/download/stsb/stsb-es.zip",
    #             "https://github.com/amyGB99/automl_benchmark/releases/download/haha/haha.zip",
    #             "https://github.com/amyGB99/automl_benchmark/releases/download/meddocan/meddocan.zip"]
    # func = ["load_paws","load_paws","load_wnli","load_wikiann","load_wikicat",
    #              "load_sst_en","load_stroke", "load_women_clothing" , "load_jobs","load_wines","load_project_kickstarter",
    #              "load_price_book","load_inferes", "load_predict_salary","load_stsb","load_stsb","load_haha","load_meddocan"]
    @classmethod
    def init(cls):
        local_path = os.path.dirname(os.path.realpath(__file__))
        list_ = os.path.join(local_path,'list_datasets.txt')
        # data = pd.DataFrame()
        # data["name"] = cls.datasets
        # data["url"] = cls.urls
        # data["func"] = cls.func
        # data.to_csv(list_,sep="\t",index= False)
        datasets_list = pd.read_csv(list_,sep="\t")
        cls.names = list(datasets_list["name"])
        cls.name_func = list(datasets_list["func"])
        cls.urls = list(datasets_list["url"])
        for i in range(len(cls.names)):
           inst = dataset.Dataset(cls.names[i],cls.urls[i],cls.return_func('.functions_load',cls.name_func[i]))  
           cls.inst[f'{cls.names[i]}'] = inst
           utils.save_dataset_definition(inst)
            
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
    def new_dataset(cls,name: str, url: str,function):
        cls.names.append(name)
        cls.urls.append(url)
        inst = dataset.Dataset(name,url,function)
        cls.inst[f'{name}'] = inst
        utils.save_dataset_definition(inst)
        cls.name_func.append(function.__name__)