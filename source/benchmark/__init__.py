
  
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

class HAutoMLBench():
    
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
    
    scoring = { 
                'accuracy' : accuracy_score, 
                'balanced_accuracy' : balanced_accuracy_score,
                
                'precision' : precision_score,
                'recall' : recall_score , 
                'f1_score': f1_score  ,
                'roc_auc' : roc_auc_score,
                
                'MAE' : mean_absolute_error,
                'MSE' : mean_squared_error,
                'RMSE' : mean_squared_error,
              }
 
    
    @classmethod
    def create_datasets(cls):
        local_path = os.path.dirname(os.path.realpath(__file__))
        info_path = os.path.join(local_path,'infos.jsonl')
        try:
            _, df = cls.__change_names(local_path, reset= True)
            infos = cls.__write_info(local_path,info_path,reset=True)
        except:
           print('Error creating benchmark datasets make sure the files: columns_type and properties , are created correctly.') 
           return   
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
        info_path = os.path.join(local_path,'infos.jsonl')
        names ,urls, funcs = cls.__read_variables_bench(local_path)
        infos = None
        if ret_info :
            infos  =jsonlines.Reader(info_path)
            with open(info_path, "r", encoding="utf-8") as file:
                infos = [line for line in jsonlines.Reader(file)]
            
        return names,urls,funcs,infos
        
    @classmethod
    def __write_info(cls,local_path,info_path,reset = False):
        dict_ = []
        properties_path = os.path.join(local_path,'properties.json')
        columns_type_path = os.path.join(local_path,'columns_types.json')
        if reset == True:
            with open(properties_path, 'r') as fp:
                properties = json.load(fp)
                with open(columns_type_path, 'r') as ff:
                    columns_type = json.load(ff)
                    for name,i in zip(properties.keys(),range(len(properties.keys()))):
                        
                        properties[name]['columns_type'] = columns_type[name]
                        dict_.append({name:properties[name]})
                         
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
                infos.pop(index) 
        elif not remove: 
            infos.append(info) 
        with jsonlines.open(info_path, 'w') as fp:
            fp.write_all(infos)
          
    @classmethod
    def __read_variables_bench(cls,local_path):
        names_path = os.path.join(local_path,'variables.tsv')
        df = pd.read_table(names_path)
        return list(df['name']),list(df['url']),list(df['func'])
   
    @classmethod
    def __change_names(cls,local_path, dataset = None, insert = False, reset =False):
         
        archive_path = os.path.join(local_path,'variables.tsv')
        
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
            "https://github.com/amyGB99/automl_benchmark/releases/download/wikineural/wikineural-en.zip",
            "https://github.com/amyGB99/automl_benchmark/releases/download/wikineural/wikineural-es.zip",
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
    def load_dataset(cls, name, format = "pandas", in_xy = True, samples = 2,encoding = 'utf-8',target = None):
        dataset = utils.load_dataset_definition(name)
        if dataset is not None:
            try:
                if target!= None:
                    if name != "google-guest":
                        if not isinstance(target,str):
                            print(f'Error: The target for {name} dataset most be only one')
                    return dataset.loader_func(dataset,format = format , in_x_y = in_xy , samples = samples, encoding = encoding,target = target)
                else:
                    return dataset.loader_func(dataset,format = format , in_x_y = in_xy , samples = samples, encoding = encoding)   
            except Exception as error :
               print(f"Error loading the dataset {name} make sure that the parameters were entered correctly and that the dataset belongs to the benchmark")
        return None
    
    @classmethod
    def load_info(cls,name):
        try:
            return utils.load_dataset_definition(name).info
        except:
            print(f"Error: There is no registered dataset with the name : {name}")
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
                        if property not in ['n_instances','n_columns', 'null_values','classes','class balance']:
                            print('Error: The feature is incorrect')
                            break
                       
                        value_property = value[property]
                        
                        if property == 'n_instances' and value_property != None :
                            value_property = value[property][0] + value[property][1]
                        
                        if min == None and max == None:
                            print('Error : Both ranges cannot be none')
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
                            print('Error: The feature is incorrect')
                            break            
                except Exception as error:
                    print(error)
                    break
            return list_        
                                         
    @classmethod
    def new_dataset(cls,name: str, url: str, function, info = None ):
        '''
        info = {'n_instances': [int,int],
                'n_columns': int , 
                'columns_type': {'name' : type},
                'targets': list[str] or str,
                'null_values': int,
                'task': str 
                'pos_label': Any,
                'classes': int, 
                'class balance':float}
        '''
        correct = False

        try:
            inst = dataset.Dataset(name,url,info,cls.return_func('.functions_load',function))
            correct = True
        except:
            try:
                inst = dataset.Dataset(name,url,info,function) 
                correct = True
            except Exception as error:
                print("Error: The load function was not entered correctly, make sure it is in the functions_load file")
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
                    'targets': None ,
                    'null_values': None,
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
                print('Error in removing the new dataset, the benchmark information was corrupted')
                print('Resetting to Benchmark Default Data.....')
                cls.__change_names(local_path,reset = True)
                cls.__write_info(local_path,info_path,reset=True)
                          
    @classmethod
    def remove_dataset(cls, name: str):
        
        local_path = os.path.dirname(os.path.realpath(__file__))
        info_path = os.path.join(local_path,'info.jsonl')
        row  ={'name': name}
        try:
            index,_ = cls.__change_names(local_path, dataset= row,insert= False,reset =False)
            cls.__update_info(info_path,remove= True,index= index)
        except Exception as error:
            print('Error in removing the new dataset, the benchmark information was corrupted')
            print('Resetting to Benchmark Default Data.....')
            cls.__change_names(local_path,reset = True)
            cls.__write_info(local_path,info_path,reset=True)

    @classmethod
    def evaluate(cls,name,y_true,y_pred, task, pos=None ,labels = None,save_path = None, name_archive ='results'):
        metric_class = [cls.scoring['accuracy'],cls.scoring['precision'],cls.scoring['recall'], 
                cls.scoring['f1_score'],cls.scoring['balanced_accuracy']] 
        
        metric_regr = [cls.scoring['MSE'],cls.scoring['MAE']]
        try:
            result_path = os.path.join(save_path,'.'.join([name_archive,'json']))
        except:    
            local_path = os.path.dirname(os.path.realpath(__file__))
            result_path = os.path.join(local_path,'.'.join([name_archive,'json']))
        results = {}   
        dict_ = {}
        try:
            with open(result_path,'r') as fp:
                results = json.load(fp)
        except:
           with open(result_path,'w') as fp:
            json.dump(dict_,fp)         
        
        if task == 'binary' or task == 'multiclass' :
            metrics =metric_class
            
        elif task == 'regression':
            metrics =metric_regr
        
        for metric in metrics:
            metric_name =metric.__name__
            try:
                if  metric_name == 'accuracy_score' or metric_name == 'balanced_accuracy_score':
                        score = metric(y_true, y_pred)      
                else:
                    if task == 'binary':
                        score1 = metric(y_true,y_pred,average = 'micro')  
                        score2 = metric(y_true,y_pred,pos_label= pos)  
                        score = {'micro': score1, 'normal': score2}
                    elif task == 'multiclass':
                        score1 = metric(y_true,y_pred,labels =labels,average = 'macro')  
                        score2 = metric(y_true,y_pred ,labels =labels,average ='weighted')    
                        score = {'macro': score1, 'weighted': score2}
                    elif task =='regression':
                        if metric_name == 'mean_squared_error': 
                            score1 = metric(y_true,y_pred,squared = False)
                            score2 = metric(y_true,y_pred)
                            score = {'RMSE': score1, 'MSE': score2}

                        else:
                            score = metric(y_true,y_pred)             
                dict_[metric_name]= score
            except Exception as error:
                print(error)
                    
        results[name] = dict_
        try:
            with open(result_path,'w') as fp:
                json.dump(results,fp,indent=4)
        except Exception as error:
            print(error)
                    
        return results[name]        