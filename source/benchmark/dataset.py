
#from importlib.resources import path
from random import sample
from yamlable import yaml_info, YamlAble
from .functions_load import *
 
import inspect
from .utils import *


@yaml_info(yaml_tag_ns="automl.benchmark")
class Dataset(YamlAble):
    def __init__(self, name: str, url: str, info, loader_func):
        #self.name, self.url, self.loader_func = name, url, loader_func
        self.name, self.url, self.info,self.loader_func = name, url, info, loader_func

        if self.loader_func is not None:
            self.loader_func_name = self.loader_func.__name__
            self.loader_func_definition = inspect.getsource(self.loader_func)

    def __repr__(self):
        """String representation for prints"""
        return f"{type(self).__name__} - { {'name': self.name, 'url': self.url, 'info': self.info, 'loader': self.loader_func_name} }"

    def __to_yaml_dict__(self):
        # Do not dump 'irrelevant'
        return {
            "name": self.name,
            "url": self.url,
            "info": self.info,
            "loader_func_name": self.loader_func_name,            
        }
    @classmethod
    def __from_yaml_dict__(cls, dct, yaml_tag):
        # Accept a default value for b
        loader = import_loader(dct["name"], dct["loader_func_name"],)
        return cls(dct["name"], dct["url"], dct['info'] , loader)
    
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
        datasets_path = os.path.join(local_path,'datasets')
        save_path = os.path.join(datasets_path, f'{self.name}')
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
     
