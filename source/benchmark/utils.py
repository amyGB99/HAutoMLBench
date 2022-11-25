import yaml
import os
import importlib
from importlib.machinery import SourceFileLoader
import jsonlines
import json
import pandas as pd


local_path = os.path.dirname(os.path.realpath(__file__))
datasets_folder_path = os.path.join(local_path,'data')

def ensure_directory(path: str):
    try:
        os.makedirs(path)
    except Exception:
        pass

    pass

#### Utils Class Dataset
def save_dataset_definition(dataset):
    save_path = os.path.join(datasets_folder_path, dataset.name)
    
    ensure_directory(save_path)
    with open(os.path.join(save_path, "dataset.yaml"), "w") as file, open(
        os.path.join(save_path, "loader.py"), "w"
    ) as file2:
        yaml.dump(dataset, file)
        file2.writelines(dataset.loader_func_definition)

def import_loader(name: str, loader_function_name: str):
    load_path = os.path.join(datasets_folder_path, name)
    loader = SourceFileLoader(
        "loader", os.path.join(load_path, "loader.py")
    ).load_module()
    return getattr(loader, loader_function_name)

def load_dataset_definition(name: str):
    load_path = os.path.join(datasets_folder_path, name)
    try:
        with open(os.path.join(load_path, "dataset.yaml"), "r") as file:
            return yaml.safe_load(file)
    except IOError as e:
        raise IOError(f"Failed to find Dataset '{name}'. {e}")


### Utils Class Benchamrk

def return_func(mod, func):
    module = importlib.import_module(mod, package="benchmark")
    function = getattr(module, func)
    return function
    
def caller_func(mod, func:str, *args):
    function = return_func(mod,func)
    function(*args)

def verify(metadata):
    try:
        keys = metadata.keys()
        if 'n_instances' not in keys or 'n_columns' not in keys  or'columns_types' not in keys or 'targets' not in keys or 'null_values' not in keys or 'task' not in keys or 'positive_class' not in keys or 'class_labels' not in keys or 'n_classes' not in keys or 'class_balance'not in keys:
            return False
        for key,value in metadata.items():
            if key =='n_columns' or  key=='null_values' or  key=='n_classes':
                if not isinstance(value, int) and value != None:  
                    return False
            elif key =='columns_types':
                if not isinstance(value, dict) and value != None:
                    return False
            elif key=='task':
                if not isinstance(value, str) and value != None:
                    return False
            elif key=='positive_class':
                if not isinstance(value, (int, str)) and value != None:
                    return False
            elif key=='class_balance':
                if isinstance(value, (float, int)) and value != None:
                    return False
            elif key=='targets':
                if not isinstance(value, (str, list)) and value != None:
                    return False    
            elif key=='class_labels':
                if not isinstance(value, list) and value != None:
                    return False
            elif key=='n_instances':
                if not isinstance(value, (list,int)) and value != None:
                    return False 
                elif isinstance(value, list):
                    for item in value:
                        if not isinstance(item,int):
                            return False    
        return True
    except:
        return False


def create_columns_type():
    labels  = [['label'], ['label'], ['label'], ['label'], ['label'],['label'],['stroke'],['Class Name'],['fraudulent'],['price'] ,['final_status'] ,['Price'], ['Label'],['salary'],['score'],['score'],['is_humor','average'],['label'],['label'],['label'],['label'],['label'],['label'],['labels'],['account_type'],['question_asker_intent_understanding','question_body_critical','question_conversational','question_expect_short_answer','question_fact_seeking','question_has_commonly_accepted_answer','question_interestingness_others','question_interestingness_self','question_multi_intent','question_not_really_a_question','question_opinion_seeking','question_type_choice','question_type_compare','question_type_consequence','question_type_definition','question_type_entity','question_type_instructions','question_type_procedure','question_type_reason_explanation','question_type_spelling','question_well_written','answer_helpful', 'answer_level_of_information','answer_plausible','answer_relevance','answer_satisfaction','answer_type_instructions','answer_type_procedure','answer_type_reason_explanation', 'answer_well_written']]
    names =["paws-x-en","paws-x-es","wnli-es","wikiann-es","wikicat-es","sst-en",
                    "stroke-prediction","women-clothing","fraudulent-jobs","spanish-wine",
                    "project-kickstarter","price-book","inferes","predict-salary","stsb-en",
                    "stsb-es","haha", "meddocan","vaccine-es","vaccine-en","sentiment-lexicons-es",
                    "wikineural-en","wikineural-es","language-identification","twitter-human-bots","google-guest"]
    all_types = {}
    for dataset,i in zip(names,range(len(names))):
        print('#####################################################')
        print(dataset)
        types = {}
        if dataset == "wikiann-es" or dataset =='meddocan' or dataset =='wikineural-es' or dataset =='wikineural-en':
            all_types[dataset] = {'tokens':'Seqtokens','tags':'Seqtags'}
            continue
        target = labels[i]
        inst= load_dataset_definition(dataset)
        all = inst.loader_func(inst,samples = 1)
        for column in all.columns:
            t = all[column].dtype.name
            if t== 'int64':
                types[column] = 'int'
            elif t== 'float64':
                types[column] = 'float'  
            elif t== 'object':
                types[column] = 'text' 
            else:
                types[column] = t       
        all_types[dataset] = types
    with open('/media/amanda/DATA1/School/Thesis/implementation/benchmark actual/automl_benchmark/source/benchmark/columns_types.json', 'w') as fp:
        json.dump(all_types, fp,indent= 4) 
        
def create_prperties():
    labels  = [['label'], ['label'], ['label'], ['label'], ['label'],['label'],['stroke'],['Class Name'],['fraudulent'],['price'] ,['final_status'] ,['Price'], ['Label'],['salary'],['score'],['score'],['is_humor'],['label'],['label'],['label'],['label'],['label'],['label'],['labels'],['account_type'],['question_asker_intent_understanding','question_body_critical','question_conversational','question_expect_short_answer','question_fact_seeking','question_has_commonly_accepted_answer','question_interestingness_others','question_interestingness_self','question_multi_intent','question_not_really_a_question','question_opinion_seeking','question_type_choice','question_type_compare','question_type_consequence','question_type_definition','question_type_entity','question_type_instructions','question_type_procedure','question_type_reason_explanation','question_type_spelling','question_well_written','answer_helpful', 'answer_level_of_information','answer_plausible','answer_relevance','answer_satisfaction','answer_type_instructions','answer_type_procedure','answer_type_reason_explanation', 'answer_well_written'],['label']]
    regres =['spanish-wine', 'price-book', 'stsb-en', 'stsb-es', 'google-guest']
    names = ["paws-x-en","paws-x-es","wnli-es","wikiann-es","wikicat-es","sst-en",
                    "stroke-prediction","women-clothing","fraudulent-jobs","spanish-wine",
                    "project-kickstarter","price-book","inferes","predict-salary","stsb-en",
                    "stsb-es","haha", "meddocan","vaccine-es","pub-health","sentiment-lexicons-es",
                    "wikineural-en","wikineural-es","language-identification","twitter-human-bots","google-guest",'trec']
    dict_ ={}
    
    for dataset,i in zip(names,range(len(names))):
        print(dataset)
        if dataset == "wikiann-es" or dataset =='meddocan' or dataset =='wikineural-es' or dataset =='wikineural-en':
          inst= load_dataset_definition(dataset)
          train,ytra,test, ytes= inst.loader_func(inst,format='pandas',in_x_y=False,samples=2)
          columns = 2
          instances = [len(train),len(test)]
          count_null = 0
          number_class = None
          balance = None
          labels_d = None
          task = 'entity'
          pos = None
          
        else:
            inst= load_dataset_definition(dataset)
            train, test= inst.loader_func(inst,format='pandas',in_x_y=False,samples=2)
            all = inst.loader_func(inst,format='pandas',in_x_y=False,samples=1)
            if dataset != "google-guest":
              target = labels[i][0]
            else:
              target = labels[i]

            columns = len(train.columns)
            instances = [len(train.axes[0]),len(test.axes[0])]
            count_null = int(all.isnull().sum().sum())
            if dataset in regres:
              number_class = None
              balance = None
              pos = None
              task = 'regression'
              labels_d = None
            else:
                number_class = len(all[target].dropna().unique())
                number_clases_train = train[target].dropna().value_counts()
                labels_d = all[target].dropna().unique().tolist()
                if number_class == 2:
                    task = 'binary'
                    if dataset =='sentiment-lexicons-es':
                      pos = 'positive'
                    elif dataset == 'twitter-human-bots':
                      pos = 'bot'
                    else:
                      pos = 1  
                else:
                    task = 'multiclass'
                    pos = None
                     
                min = number_clases_train.min()
                max = number_clases_train.max()
                balance = min/max 
        dict_[dataset] = {'n_instances':instances,
                          'n_columns':columns,
                          'targets':target,
                          'null_values':count_null,
                          'class_labels': labels_d,
                          'positive_class': pos,
                          'n_classes': number_class,
                          'task': task,
                          'class_balance':balance}
    
    with open('/media/amanda/DATA1/School/Thesis/implementation/benchmark actual/automl_benchmark/source/benchmark/properties.json', 'w') as fp:
        json.dump(dict_, fp,indent= 4,ensure_ascii= False)    
                     
def get_properties (ret_info = False):
    info_path = os.path.join(local_path,'datasets_info.jsonl')
    names ,urls, funcs = read_variables_bench()
    infos = None
    if ret_info :
        infos  =jsonlines.Reader(info_path)
        with open(info_path, "r", encoding="utf-8") as file:
            infos = [line for line in jsonlines.Reader(file)]  
    return names, urls, funcs , infos
        
def read_variables_bench():
    variables_path = os.path.join(local_path,'variables.tsv')
    df = pd.read_table(variables_path)
    return list(df['name']),list(df['url']),list(df['func'])

def init_variables_file():

    variables_path = os.path.join(local_path,'variables.tsv')
    
    datasets_name = ["paws-x-en","paws-x-es","wnli-es","wikiann-es","wikicat-es","sst-en",
                "stroke-prediction","women-clothing","fraudulent-jobs","spanish-wine",
                "project-kickstarter","price-book","inferes","predict-salary","stsb-en",
                "stsb-es","haha", "meddocan","vaccine-es","pub-health","sentiment-lexicons-es",
                "wikineural-en","wikineural-es","language-identification","twitter-human-bots","google-guest","trec"]
    
    datasets_urls = ["https://github.com/amyGB99/automl_benchmark/releases/download/paws-x/paws-x-en.zip",
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
        "https://github.com/amyGB99/automl_benchmark/releases/download/pub-health/pub-health.zip",
        "https://github.com/amyGB99/automl_benchmark/releases/download/sentiment-lexicons/sentiment-lexicons-es.zip",
        "https://github.com/amyGB99/automl_benchmark/releases/download/wikineural/wikineural-en.zip",
        "https://github.com/amyGB99/automl_benchmark/releases/download/wikineural/wikineural-es.zip",
        "https://github.com/amyGB99/automl_benchmark/releases/download/language-identification/language-identification.zip",
        "https://github.com/amyGB99/automl_benchmark/releases/download/twitter-human-bots/twitter-human-bots.zip",
        "https://github.com/amyGB99/automl_benchmark/releases/download/google-guest/google-guest.zip",
        "https://github.com/amyGB99/automl_benchmark/releases/download/trec/trec.zip"]

    datasets_func = ["load_paws","load_paws","load_wnli","load_wikiann","load_wikicat",
                "load_sst_en","load_stroke", "load_women_clothing" , "load_jobs","load_wines","load_project_kickstarter",
                "load_price_book","load_inferes", "load_predict_salary","load_stsb","load_stsb","load_haha","load_meddocan",
                "load_vaccine","load_pub_health","load_sentiment","load_wikineural","load_wikineural","load_language",
                "load_twitter_human","load_google_guest","load_trec"]
    
    df = pd.DataFrame()
    df['name'] = datasets_name
    df['url'] = datasets_urls
    df['func'] = datasets_func
    
    df.to_csv(variables_path, sep='\t',index=False)
    
    return df 
    
def update_variables_file(dataset: dict, operation = 'insert'):
    '''
    input:
        dataset = { name: str , url: str :  fuction: str}
        operation = str : bool : 'insert' or 'remove'
    output: index of the dataset that was removed or added  
    '''
        
    variables_path = os.path.join(local_path,'variables.tsv')
    
    df = pd.read_table(variables_path)
    names = df['name'].tolist()
    
    try:
        index = names.index(dataset['name'])   
    except:
        index = None
    if dataset is not None:
        if index != None:
            if operation == 'insert':
                df.iloc[index,:] = dataset
            else:
                df = df.drop([index],axis=0).reset_index(drop=True)         
        else:    
            if operation == 'insert':
                df = df.append(dataset,ignore_index = True)
    
    df.to_csv(variables_path, sep='\t',index =False) 
    return index 

def init_information_file():
    '''
    output : information: [dict]
    
    '''
    
    informations = []
    datasets_info_path = os.path.join(local_path,'datasets_info.jsonl')
    properties_path = os.path.join(local_path,'properties.json')
    columns_types_path = os.path.join(local_path,'columns_types.json')
    
    with open(properties_path, 'r') as fp:
        properties = json.load(fp)
        with open(columns_types_path, 'r') as ff:
            columns_type = json.load(ff)
            for name,i in zip(properties.keys(),range(len(properties.keys()))):
                properties[name]['columns_types'] = columns_type[name]
                informations.append({name:properties[name]})
                        
        with jsonlines.open(datasets_info_path, 'w') as fp:
            fp.write_all(informations)
    return informations
 
def update_information_file(metadata = None, operation = 'insert', index= None):
    '''
        input:
        metadata = { name:{'n_instances': [int,int] or None,
                            'n_columns': int or None, 
                            'columns_types': {'name' : type} or None,
                            'targets': list[str] or str or None,
                            'null_values': int or None,
                            'task': str or None,
                            'positive_class': Any,
                            'class_labels': [Any],
                            'n_classes': int or None, 
                            'class_balance':float or None}}
                            
        operation = str : bool : 'insert' or 'remove'
        index = int or None, index where to insert or remove
        
    '''
    datasets_info_path = os.path.join(local_path,'datasets_info.jsonl')
    
    with open(datasets_info_path, "r", encoding="utf-8") as file:
        informations = [line for line in jsonlines.Reader(file)]
    
    if  index!= None:
        if operation == 'insert':
            informations[index] = metadata    
        else:
            informations.pop(index) 
    elif operation == 'insert': 
        informations.append(metadata) 
    with jsonlines.open(datasets_info_path, 'w') as fp:
        fp.write_all(informations)
          
### Metrics _entity       
def get_qvals(y, predicted):
    tp = 0
    fp = 0
    fn = 0
    total_sentences = 0
    for i in range(len(y)):
        for j in range(len(y[i])):
            tag = y[i][j]
            predicted_tag = predicted[i][j]

            if tag != "O" or tag != 0:
                if tag == predicted_tag:
                    tp += 1
                else:
                    fn += 1
            elif tag != predicted_tag:
                fp += 1
        total_sentences += 1

    return tp, fp, fn, total_sentences

def precision(y, predicted):
    """
    precision evaluation function from [MEDDOCAN iberleaf 2018](https://github.com/PlanTL-SANIDAD/SPACCC_MEDDOCAN)
    """
    tp, fp, fn, total_sentences = get_qvals(y, predicted)
    try:
        return tp / float(tp + fp)
    except ZeroDivisionError:
        return 0.0

def recall(y, predicted):
    """
    recall evaluation function from [MEDDOCAN iberleaf 2018](https://github.com/PlanTL-SANIDAD/SPACCC_MEDDOCAN)
    """
    tp, fp, fn, total_sentences = get_qvals(y, predicted)
    try:
        return tp / float(tp + fn)
    except ZeroDivisionError:
        return 0.0

def F1_beta(y, predicted, beta=1):
    """
    F1 evaluation function from [MEDDOCAN iberleaf 2018](https://github.com/PlanTL-SANIDAD/SPACCC_MEDDOCAN)
    """
    p = precision(predicted, y)
    r = recall(predicted, y)
    try:
        return (1 + beta ** 2) * ((p * r) / (p + r))
    except ZeroDivisionError:
        return 0.0
        pass
