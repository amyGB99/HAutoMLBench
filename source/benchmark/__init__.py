
  
from numpy import average
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score,recall_score,precision_score,balanced_accuracy_score,mean_squared_error,mean_absolute_error
from sklearn.metrics  import make_scorer
from yaml import dump
from benchmark import dataset
from benchmark import utils
from benchmark import functions_load
import os
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
                'f1_beta': utils.F1_beta,
                'entity_precision': utils.precision,
                'entity_recall':utils.recall
                
              }
 
    
    @classmethod
    def init(cls):
        try:
            variables = utils.init_variables_file()
            informations = utils.init_information_file()
        except:
           print('Error creating benchmark datasets make sure the files: columns_type and properties, are created correctly.') 
           return   
        datasets_names = variables['name'].to_list()
        datasets_urls = variables['url'].to_list()
        datasets_funcs = variables['func'].to_list()
        for i in range(len(datasets_names)):
            try:
                dataset_name = datasets_names[i]
                dataset_metadata = informations[i][dataset_name]
                dataset_instance = dataset.Dataset(dataset_name, datasets_urls[i], dataset_metadata, utils.return_func('.functions_load',datasets_funcs[i]))  
                utils.save_dataset_definition(dataset_instance)
            except Exception as error:
                print(error)
            
    @classmethod
    def get_dataset(cls, name):
        return utils.load_dataset_definition(name)
   
    @classmethod
    def filter(cls, task = None,expresion = None):
        '''
        task: str = 'binary', 'regression,'multiclass'
        expression: tuple(len(3)) = (property,min,max) : min <= property < max
        
        return : list[str]
        '''  
        datasets_names,_,_,datasets_infos = utils.get_properties(True)
        if task == None and expresion == None:
            return datasets_names
        else:
            datasets = []
            for dataset_info,i in zip(datasets_infos,range(len(datasets_names))):
                dataset_name = datasets_names[i]
                metadata = dataset_info[dataset_name]
                if metadata is None:
                    continue
                try:
                    if expresion == None:
                        if task == metadata['task']:
                            datasets.append(dataset_name) 
                        continue
                    else:
                        if task != None and task != metadata['task']:
                            continue
                        property,min,max = expresion
                        if property not in ['n_instances','n_columns', 'null_values','classes','class balance']:
                            print(f'Error: The entered property {property} is incorrect')
                            break
                        if not isinstance(min,(int,float)) and not isinstance(max,(int,float)):
                            print('Error : Both ranges in the expression cannot be None')
                            break
                        value_property = metadata[property]
                        
                        if value_property is None:
                            continue
                        if property == 'n_instances':
                            if isinstance(value_property,list):
                                value_property = 0  
                                for value in  metadata[property]:
                                    value_property += value   
                        if min is not None and max is not None :
                            if value_property >= min and value_property < max:
                                datasets.append(dataset_name)
                        elif min is None:
                            if value_property < max:
                                datasets.append(dataset_name)
                        else:
                            if value_property >= min:
                                datasets.append(dataset_name)                    
                except Exception as error:
                    print(error)
                    break
            return datasets        
                                         
    @classmethod
    def add_dataset(cls, name: str, url: str, function, metadata = None ):
        '''
        info = {'n_instances': [int,int],
                'n_columns': int , 
                'columns_types': {'name' : type},
                'targets': list[str] or str,
                'null_values': int,
                'task': str 
                'positive_class': Any,
                'class_labels': [Any] 
                'n_classes': int, 
                'class_balance':float}
        '''
        try:
            dataset_instance = dataset.Dataset(name, url, metadata, function) 
            new_row = {'name': name, 'url': url,'func': dataset_instance.loader_func_name}
            if metadata is None:
                metadata = { name:{'n_instances': None, 
                    'n_columns': None, 
                    'columns_types': None,
                    'targets': None ,
                    'null_values': None,
                    'task': None,
                    'positive_class': None,
                    'class_labels': None,
                    'n_classes': None, 
                    'class_balance':None}
                }
            else:
                
                if not utils.verify(metadata):
                    print("The metadata of the new set was entered incorrectly")
                    return
                metadata = { name : metadata}   
            try:
                index = utils.update_variables_file(dataset = new_row, operation= 'insert')
                utils.update_information_file(metadata = metadata, operation= 'insert', index = index)   
                utils.save_dataset_definition(dataset_instance)
            except:
                print('Error in adding the new dataset, the benchmark information was corrupted')
                print('Resetting to Benchmark Default Data.....')
                utils.init_variables_file()
                utils.init_information_file()
                                              
        except:
            print("Error: The function parameter was entered incorrectly, make sure it is a python function")

    @classmethod
    def remove_dataset(cls, name: str):
        row  ={'name': name}
        try:
            index = utils.update_variables_file(dataset= row, operation= 'remove')
            utils.update_information_file(operation= 'remove',index= index)
        except Exception as error:
            print('Error in removing the new dataset, the benchmark information was corrupted')
            print('Resetting to Benchmark Default Data.....')
            utils.init_variables_file()
            utils.init_information_file()

    @classmethod
    def evaluate(cls,name,y_true,y_pred, is_multilabel = False,task =None, positive_class = None , class_labels = None, save_path = None, name_archive ='results'):
        if len(y_true)!= len(y_pred):
            print("Error: The predictions and the true target must have the same shape and length")
            return None
        metric_class = [cls.scoring['accuracy'],cls.scoring['precision'],cls.scoring['recall'], 
                cls.scoring['f1_score'],cls.scoring['balanced_accuracy']] 
        
        metric_regr = [cls.scoring['MSE'],cls.scoring['MAE']]
        
        metric_entity = [cls.scoring['f1_beta'], cls.scoring['entity_precision'],cls.scoring['entity_recall']]
        
        try:
            result_path = os.path.join(save_path,'.'.join([name_archive,'json']))
        except:    
            local_path = os.path.dirname(os.path.realpath(__file__))
            result_path = os.path.join(local_path,'.'.join([name_archive,'json']))
        results = {}   
        dict_ = {}
        metadata = None
        try:
            instance = cls.get_dataset(name)
        except:
            instance = None    
        is_multilabel = is_multilabel if name == 'google-guest' and is_multilabel is not None else False
        if instance is not None:
            metadata = instance.info

        if metadata is not None and task is None:
            task = metadata['task']
        if metadata is not None and positive_class is None:
            positive_class = metadata['positive_class']
        if metadata is not None and class_labels is None:
            class_labels = metadata['class_labels']
                
        try:
            with open(result_path,'r') as fp:
                results = json.load(fp)
        except:
           with open(result_path,'w') as fp:
            json.dump(dict_,fp)         
        
        if task is None:
            print('The task type of the dataset to evaluate cannot be None')
            return None
        if task == 'binary':
            metrics = metric_class
            if positive_class is None:
                print('If the task type is Binary to evaluate the dataset the positive_class cannot be None')
                return None
        elif task == 'multiclass':
            metrics = metric_class
            if class_labels is None:
                print('If the task type is Multiclass to evaluate the dataset the class_labels cannot be None')
                return None
        elif task == 'regression':
            metrics = metric_regr
        elif task =='entity':
            metrics = metric_entity
            
        for metric in metrics:
            metric_name =metric.__name__
            try:
                if  metric_name == 'accuracy_score' or metric_name == 'balanced_accuracy_score' or metric_name == 'precision' or metric_name == 'recall' or metric_name == 'F1_beta' :
                        score = metric(y_true, y_pred)      
                else:
                    
                    if task == 'binary':
                        score1 = metric(y_true,y_pred, average = 'micro')  
                        score2 = metric(y_true,y_pred,pos_label= positive_class)  
                        score = {'micro': score1, 'normal': score2}
                    elif task == 'multiclass':
                        score1 = metric(y_true,y_pred,labels = class_labels,average = 'macro')  
                        score2 = metric(y_true,y_pred ,labels =class_labels,average ='weighted')    
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