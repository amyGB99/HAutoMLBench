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
    # self.datasets = ["paws-x-en","paws-x-es","wnli-es","wikiann-es","wikicat-es","sst-en",
    #                  "stroke-prediction","women-clothing","fraudulent-jobs","spnish-wine",
    #                  "project-kickstarter","price-book","inferes","predict-salary","stsb-en",
    #                  "stsb-es","haha", "mendocam"]
    # self.urls = ["https://github.com/amyGB99/automl_benchmark/releases/download/paws-x/paws-x-en.zip",
    #             "https://github.com/amyGB99/automl_benchmark/releases/download/paws-x/paws-x-es.zip",
    #             "https://github.com/autogoal/benchmark/releases/download/wnli-es/wnli-es.zip",
    #             "https://github.com/amyGB99/automl_benchmark/releases/download/wikiann-es/wikiann-es.zip",
    #             "https://github.com/amyGB99/automl_benchmark/releases/download/wikiann-es/wikiann-es.zip",
    #             "https://github.com/autogoal/benchmark/releases/download/sst-en/sst-en.zip",
    #             "https://github.com/autogoal/benchmark/releases/download/stroke-prediction/stroke-prediction.zip",
    #             "https://github.com/autogoal/benchmark/releases/download/women-clothing/women-clothing.zip",
    #             "https://github.com/autogoal/benchmark/releases/download/fraudulent-jobs/fraudulent-jobs.zip",
    #             "https://github.com/autogoal/benchmark/releases/download/wine/spnish-wine.zip",
    #             "https://github.com/autogoal/benchmark/releases/download/project-kickstarter/project-kickstarter.zip",
    #             "https://github.com/autogoal/benchmark/releases/download/price-book/price-book.zip",
    #             "https://github.com/autogoal/benchmark/releases/download/paws-x/paws-x-en.zip",
    #             "https://github.com/autogoal/benchmark/releases/download/paws-x/paws-x-en.zip",
    #             "https://github.com/autogoal/benchmark/releases/download/paws-x/paws-x-en.zip",
    #             "https://github.com/autogoal/benchmark/releases/download/paws-x/paws-x-en.zip",
    #             "https://github.com/autogoal/benchmark/releases/download/paws-x/paws-x-en.zip",
    #             "https://github.com/autogoal/benchmark/releases/download/paws-x/paws-x-en.zip"]
    # self.func = ["load_two_sentences","load_two_sentences","load_two_sentences","load_wikiann","load_wikicat",
    #              "load_sst_en","load_stroke", "load_women_clothing" , "load_jobs","load_wines","load_project_kickstarter",
    #              "load_price_book","load_price_book", "load_predict_salary","load_stsb_en","load_stsb_es","load_wines","load_wines"]
    @classmethod
    def init(cls):
        local_path = os.path.dirname(os.path.realpath(__file__))
        list_ = os.path.join(local_path,'list_datasets.txt')
        # data = pd.DataFrame()
        # data["name"] = self.datasets
        # data["url"] = self.urls
        # data["func"] = self.func
        # data.to_csv(list,sep="\t",index= False)
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
    def load_dataset(cls, name, format ="panda", in_xy = True, samples = 2):
        dataset = utils.load_dataset_definition(name)
        return dataset.loader_func(cls.inst[name],format , in_xy , samples)
    
    @classmethod
    def new_dataset(cls,name: str, url: str,function):
        cls.names.append(name)
        cls.urls.append(url)
        inst = dataset.Dataset(name,url,function)
        cls.inst[f'{name}'] = inst
        utils.save_dataset_definition(inst)
        cls.name_func.append(function.__name__)
        
    
        
        
        

#if __name__ == '__main__':
 #   caller('functions', 'f1', 10, 20, name='Cesar', language='Python')

#a = AutoMLBench()
#s = a.load_dataset('paws-x-en',format ="panda", in_xy = False, samples = 1)
#print(s)
