
#from importlib.resources import path
from random import sample
from yamlable import yaml_info, YamlAble
from functions_load import *
 
import inspect
import utils


@yaml_info(yaml_tag_ns="automl.benchmark")
class Dataset(YamlAble):
    def __init__(self, name: str, url: str, loader_func=None):
        self.name, self.url, self.loader_func = name, url, loader_func

        if self.loader_func is not None:
            self.loader_func_name = self.loader_func.__name__
            self.loader_func_definition = inspect.getsource(self.loader_func)

    def __repr__(self):
        """String representation for prints"""
        return f"{type(self).__name__} - { {'name': self.name, 'url': self.url, 'loader': self.loader_func_name} }"

    def __to_yaml_dict__(self):
        # Do not dump 'irrelevant'
        return {
            "name": self.name,
            "url": self.url,
            "loader_func_name": self.loader_func_name,            
        }
    @classmethod
    def __from_yaml_dict__(cls, dct, yaml_tag):
        # Accept a default value for b
        loader = utils.import_loader(dct["name"], dct["loader_func_name"],)
        return cls(dct["name"], dct["url"], loader)
    
    def download(self):
        '''
        Download the dataset and unzip it. Returns the path of the decompression file
        
        Return  
        file : str 
        '''
        import wget 
        import os
        import shutil
    
        local_path = os.path.dirname(os.path.realpath(__file__))
        save_path = os.path.join(local_path, self.name)
        zip = os.path.join(save_path,f'{self.name}.zip')
        file = os.path.join(save_path,f'{self.name}')
        if not os.path.exists(zip) and not os.path.isfile(file) :
            try:
                wget.download(self.url, save_path) 
            except:
                print("Download failed please try again")        
        if os.path.exists(zip) and not os.path.isfile(file):
            shutil.unpack_archive(zip, save_path)   
        return file    
     
# USAGE EXAMPLE #####
#
def fone(url):
    print("1 " + url)

# def ftwo(url):
#     print("2 " + url)

#a = Dataset("paws-x-en", "https://github.com/autogoal/benchmark/releases/download/paws-x/paws-x-es.zip", load_two_sentences )
#a = Dataset("wnli-es", "https://github.com/autogoal/benchmark/releases/download/wnli-es/wnli-es.zip", load_wnli)
#a = Dataset("wikiann-es", "https://github.com/amyGB99/automl_benchmark/releases/download/wikiann-es/wikiann-es.zip", load_wikiann)
#a = Dataset("wikicat-es", "https://github.com/amyGB99/automl_benchmark/releases/download/wikiann-es/wikiann-es.zip", load_wikicat)
#a = Dataset("sst-en", "https://github.com/autogoal/benchmark/releases/download/sst-en/sst-en.zip", load_sst_en)
#a = Dataset("stroke-prediction", "https://github.com/autogoal/benchmark/releases/download/stroke-prediction/stroke-prediction.zip", load_stroke)
#a = Dataset("women-clothing", "https://github.com/autogoal/benchmark/releases/download/women-clothing/women-clothing.zip", load_women_clothing)
#a = Dataset("fraudulent-jobs", "https://github.com/autogoal/benchmark/releases/download/fraudulent-jobs/fraudulent-jobs.zip", load_jobs)
#a = Dataset("spnish-wine", "https://github.com/autogoal/benchmark/releases/download/wine/spnish-wine.zip", load_wines)
#a = Dataset("project-kickstarter", "https://github.com/autogoal/benchmark/releases/download/project-kickstarter/project-kickstarter.zip", load_project_kickstarter)
#a = Dataset("price-book", "https://github.com/autogoal/benchmark/releases/download/price-book/price-book.zip", load_price_book)
a = Dataset("paws-x-es", "https://github.com/autogoal/benchmark/releases/download/paws-x/paws-x-en.zip", load_two_sentences)
#a = Dataset("inferes", "https://github.com/autogoal/benchmark/releases/download/paws-x/paws-x-en.zip", load_inferes)
#a = Dataset("predict-salary", "https://github.com/autogoal/benchmark/releases/download/paws-x/paws-x-en.zip", load_predict_salary)
#a = Dataset("stsb-en", "https://github.com/autogoal/benchmark/releases/download/paws-x/paws-x-en.zip", load_stsb_en)
#a = Dataset("stsb-es", "https://github.com/autogoal/benchmark/releases/download/paws-x/paws-x-en.zip", load_stsb_es)

#a = Dataset("1", "<this is an url>", fone)
# b = Dataset("2", "<this is another url>", ftwo)
#
utils.save_dataset_definition(a)
# utils.save_dataset_definition(b)
#
#c = utils.load_dataset_definition("project-kickstarter")
d = utils.load_dataset_definition("paws-x-es")
#d.loader_func(a)
#X_train,y_train,X_test,y_test = d.loader_func(a,format = "list")
#X_train,y_train,X_test,y_test = d.loader_func(a)
#X,y = d.loader_func(a,samples = 1)
#dataset = d.loader_func(a,in_x_y= False, samples = 1)
train,test = d.loader_func(a,in_x_y= False, samples = 2)

#print(dataset)
#print(ytr)
# print(len(yde))

print("##########################################")
#print(c, c.loader_func_definition, sep="\n")
# print(d, d.loader_func_definition, sep="\n")


