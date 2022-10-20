
######################## Text #################################


#paws-x-es,paws-x-en,wnli
def load_two_sentences(self,format = "pandas",in_x_y = True ,samples =  2):
    '''
    Return 
    Train 
    X_tr : list (list(string)) : Strings are sentences
    y_tr : list(int)
    Validation 
    X_de : list (list(string)) : Strings are sentences
    y_de : list(int)
    Test
    X_te : list (list(strings)) Strings are sentences
    y_te : list(int)
    
    '''
    import pandas as pd 
    import os 
    path = self.download()

    train = os.path.join(path,'train.txt')
    test =  os.path.join(path,'test.txt')
    all = os.path.join(path,'all.txt')
    
    # dftr = pd.read_table(train)
    # dfte = pd.read_table(test)
    # dfde = pd.read_table(dev)
    # dfall = 
    dftr = pd.read_csv(train,sep="\t")
    dfte = pd.read_csv(test,sep="\t")
    dfall = pd.read_csv(all,sep="\t")
    if in_x_y == False:
        if samples == 1:
            return dfall.drop(['id'],axis=1)
        elif samples ==2:
            return dftr.drop(['id'],axis=1),dfte.drop(['id'],axis=1)
        else:
            print("Incorrect params") 
    else:
        dummy_tr = dftr.filter(regex='(sentence1|sentence2)') 
        y_tr = dftr.filter(regex='label')

        dummy_te = dfte.filter(regex='(sentence1|sentence2)') 
        y_te = dfte.filter(regex='label') 
        
        if samples== 1:
            dummy_all = dfall.filter(regex='(sentence1|sentence2)') 
            y = dfall.filter(regex='label')
            if format == "pandas":
                return dummy_all,y
            else:
                X = dummy_all.to_numpy().tolist()
                return X,list(y['label']) 
        elif samples ==2:
            if format == "pandas":
                return dummy_tr,y_tr,dummy_te,y_te 
            else:
                X_tr = dummy_tr.to_numpy().tolist()
                X_te = dummy_te.to_numpy().tolist()
                return X_tr, list(y_tr['label']),X_te, list(y_te['label'])  
        else:
            print("Incorrect params")    
    
def load_wikiann(self, format = "pandas", samples =3):#falta
    '''
    Return :
     dict:   {
        'tokens': ["rick", "and", "morty", "are", "cool", "."],
        'ner_tags': ["B-PER", "O" , "B-PER", "O", "O", "O"],
        'langs': ["en", "en", "en", "en", "en", "en"]
        'spans': ["PER: rick", "PER: morty"]
        }
    '''
    import pandas as pd 
    import os 
    import json
    
    path = self.download()
    train = os.path.join(path,'train.json')
    dev =  os.path.join(path,'dev.json')
    test =  os.path.join(path,'test.json')
    
    def generate_examples(filepath):
        
        with open(filepath, encoding="utf-8") as f:
            wikiann_es = json.load(f)
            for id_, data in enumerate(wikiann_es["data"]):
                tokens = data["tokens"]
                ner_tags = data["ner_tags"]
                spans = data["spans"]
                langs = data["langs"]
                yield {"langs":langs,"tokens": tokens,"ner_tags":ner_tags,"spans":spans}
    
    Xtr = []
    ytr = []
    
    Xte = []
    yte = []
    yde = []
    Xde = []
    
    for item in generate_examples(train):
        Xtr.append(item['tokens'])
        ytr.append(item['ner_tags'])
    for item in generate_examples(dev):
        Xde.append(item['tokens'])
        yde.append(item['ner_tags'])
    for item in generate_examples(test):
        Xte.append(item['tokens'])
        yte.append(item['ner_tags'])       
    return Xtr,ytr, Xde, yde, Xte,yte  


#wikicat, sst-en
def load_wikicat(self, format = "pandas", samples =3):
    import pandas as pd 
    import os 
    import numpy as np
    path = self.download()
    
    ptrain = os.path.join(path,'train.txt')
    pdev = os.path.join(path,'dev.txt')
    ptest = os.path.join(path,'test.txt')
    
    dftr = pd.read_csv(ptrain,sep="\t")
    
    dfde = pd.read_csv(pdev,sep="\t")
    
    dfte = pd.read_csv(ptest,sep="\t")
    
    
    # ptrain = os.path.join(path,'train.tsv')
    # pdev =  os.path.join(path,'dev.tsv')
    # ptest =  os.path.join(path,'test.tsv')
    
    # dftr = pd.read_table(ptrain)
    # dfte = pd.read_table(ptest)
    # dfde = pd.read_table(pdev)

    y_tr = list(dftr['label'])
    X_tr = list(dftr['text'])

    y_de = list(dfde['label'])
    X_de = list(dfde['text'])
    
    y_te = list(dfte['label'])
    X_te = list(dfte['text'])
    
    return X_tr,y_tr, X_de, y_de, X_te,y_te  
 
def load_sst_en(self, format = "pandas", samples =3):
    import pandas as pd
    import os 
    path = self.download()
    ptrain = os.path.join(path,'sst_train.txt')
    pdev = os.path.join(path,'sst_dev.txt')
    ptest = os.path.join(path,'sst_test.txt')
    
    dftr = pd.read_csv(ptrain,sep="\t",header=None,names=['label','text'])
    dftr['label'] = dftr['label'].str.replace("__label__","").astype('int')
    
    dfde = pd.read_csv(pdev,sep="\t",header=None,names=['label','text'])
    dfde['label'] = dfde['label'].str.replace("__label__","").astype('int')
    
    dfte = pd.read_csv(ptest,sep="\t",header=None,names=['label','text'])
    dfte['label'] = dfte['label'].str.replace("__label__","").astype('int')

    #ptrain = os.path.join(path,'train.tsv')
    # pdev = os.path.join(path,'dev.tsv')
    # ptest = os.path.join(path,'test.tsv')
    
    # dftr = pd.read_table(ptrain)
    # dfte = pd.read_table(ptest)
    # dfde = pd.read_table(pdev)
    
    Xtr= list(dftr['text'])
    ytr = list(dftr['label'])
    
    Xde = list(dfde['text'])
    yde = list(dfde['label'])
    
    
    Xte = list(dfte['text'])
    yte = list(dfte['label'])

    return Xtr,ytr,Xde,yde,Xte,yte
    
def load_inferes(self, format = "pandas", samples =3):
    import pandas as pd
    import os 
    path = self.download()
    train = os.path.join(path,'train_.csv')
    dev =  os.path.join(path,'dev.csv')
    test =  os.path.join(path,'test.csv')
    dftr = pd.read_csv(train)
    dfte = pd.read_csv(test)
    dfde = pd.read_csv(dev)
    dummy_tr = dftr.filter(regex='(Premise|Hypothesis|Topic|Anno|Anno_Type)') 
    y_tr = dftr['Label'].to_numpy().tolist() 
    X_tr = dummy_tr.to_numpy().tolist()

    dummy_de = dfde.filter(regex='(Premise|Hypothesis|Topic|Anno|Anno_Type)') 
    y_de = dfde['Label'].to_numpy().tolist()
    X_de = dummy_de.to_numpy().tolist()
    
    dummy_te = dfte.filter(regex='(Premise|Hypothesis|Topic|Anno|Anno_Type)') 
    y_te = dfte['Label'].to_numpy().tolist()
    X_te = dummy_te.to_numpy().tolist()
    return X_tr,y_tr, X_de, y_de, X_te,y_te  
 
def load_stsb_en(self, format = "pandas", samples =3):
    import pandas as pd
    import os 
    path = self.download()
    
    train = os.path.join(path,'stsb-en-train.csv')
    dev =  os.path.join(path,'stsb-en-dev.csv')
    test =  os.path.join(path,'stsb-en-test.csv')
    dftr = pd.read_csv(train, names=['sentence1','sentence2','score'])
    dfde = pd.read_csv(dev, names=['sentence1','sentence2','score'])
    dfte = pd.read_csv(test ,names=['sentence1','sentence2','score'])
    
    dummy_tr = dftr.filter(regex='(sentence1|sentence2)') 
    y_tr = dftr['score'].to_numpy().tolist() 
    X_tr = dummy_tr.to_numpy().tolist()

    dummy_de = dfde.filter(regex='(sentence1|sentence2)') 
    y_de = dfde['score'].to_numpy().tolist()
    X_de = dummy_de.to_numpy().tolist()
    
    dummy_te = dfte.filter(regex='(sentence1|sentence2)') 
    y_te = dfte['score'].to_numpy().tolist()
    X_te = dummy_te.to_numpy().tolist()
   
    return X_tr,y_tr, X_de, y_de, X_te,y_te  

def load_stsb_es(self, format = "pandas", samples =3):
    import pandas as pd
    import os 
    path = self.download()
    train = os.path.join(path,'stsb-es-train.csv')
    dev =  os.path.join(path,'stsb-es-dev.csv')
    test =  os.path.join(path,'stsb-es-test.csv')
    dftr = pd.read_csv(train, names=['sentence1','sentence2','score'])
    dfde = pd.read_csv(dev, names=['sentence1','sentence2','score'])
    dfte = pd.read_csv(test ,names=['sentence1','sentence2','score'])
    
    dummy_tr = dftr.filter(regex='(sentence1|sentence2)') 
    y_tr = dftr['score'].to_numpy().tolist() 
    X_tr = dummy_tr.to_numpy().tolist()

    dummy_de = dfde.filter(regex='(sentence1|sentence2)') 
    y_de = dfde['score'].to_numpy().tolist()
    X_de = dummy_de.to_numpy().tolist()
    
    dummy_te = dfte.filter(regex='(sentence1|sentence2)') 
    y_te = dfte['score'].to_numpy().tolist()
    X_te = dummy_te.to_numpy().tolist()

    return X_tr,y_tr, X_de, y_de, X_te,y_te  


################## Multimodales  ###########################

def load_predict_salary(self, format = "pandas", samples =3):
    import pandas as pd
    import os 
    path = self.download()
    ptrain = os.path.join(path,'train.csv')
    pdev =  os.path.join(path,'dev.csv')
    ptest =  os.path.join(path,'test.csv')
    
    train = pd.read_csv(ptrain)
    test = pd.read_csv(ptest)
    dev = pd.read_csv(pdev)

    y_tr = train['salary'].to_numpy().tolist()
    X_tr= train.drop(['salary'], axis=1).to_numpy().tolist()
    
    y_te = test['salary'].to_numpy().tolist()
    X_te= test.drop(['salary'], axis=1).to_numpy().tolist()
    
    
    y_de = dev['salary'].to_numpy().tolist()
    X_de= dev.drop(['salary'], axis=1).to_numpy().tolist()
    return X_tr,y_tr,X_de,y_de ,X_te,y_te

def load_price_book(self, format = "pandas", samples =3):   
    '''
        Return 
        Train 
        X_tr : list (list(string)) : 8 strings 
        y_tr : list(int)
        Validation 
        X_de : list (list(string)) : 8 strings 
        y_de : list(int)
        Test
        X_te : list (list(string)) 8 :string
        y_te : list(int)
        
     '''
        
    import pandas as pd 
    import os 
    path = self.download()
    # Load the xlsx file

    ptrain = os.path.join(path,'train.xlsx')
    ptest = os.path.join(path,'test.xlsx')
    # df_te["Class Name"] = df_te["Class Name"].astype("category")
    # print(df_te['Class Name'])
    # df_te = df_te[df_te["Class Name"].notna()]
    # print(df_te.isnull().sum())
    
    train = pd.read_excel(ptrain)
    dev = pd.read_excel(pdev)
    test = pd.read_excel(ptest)

    # Read the values of the file in the dataframe
    df_train = pd.DataFrame(train)
    df_dev = pd.DataFrame(dev)
    df_test = pd.DataFrame(test)
    
    y_tr = df_train['Price'].to_numpy().tolist()
    X_tr= df_train.drop(['Price'], axis=1).to_numpy().tolist()
    
    y_de = df_dev['Price'].to_numpy().tolist()
    X_de= df_dev.drop(['Price'], axis=1).to_numpy().tolist()
    
    y_te = df_test['Price'].to_numpy().tolist()
    X_te= df_test.drop(['Price'], axis=1).to_numpy().tolist()
    
    
    return X_tr,y_tr,X_de,y_de ,X_te,y_te

def load_stroke(self, format = "pandas", samples =3):
    '''
        Return 
        Train 
        X_tr : list (list(features)) 
        y_tr : list(int)
        Validation 
        X_de : list (list(features)) 
        y_de : list(int)
        Test
        X_te : list (list(features))
        y_te : list(int)
        features: 
        gender : object, age: float64, hypertension: int64, heart_disease: int64
        ever_married: object,work_type: object, Residence_type: object, avg_glucose_level: float64
        bmi: float64, smoking_status: object
     '''
    import pandas as pd  
    import os 
    path = self.download()
    ptrain = os.path.join(path,'train.csv')
    pdev =  os.path.join(path,'dev.csv')
    ptest = os.path.join(path,'test.csv')
    
    df_tr = pd.read_csv(ptrain)
    df_de = pd.read_csv(pdev)
    df_te = pd.read_csv(ptest)
    
    y_tr = df_tr['stroke'].to_numpy().tolist()
    y_de = df_de['stroke'].to_numpy().tolist()
    y_te = df_te['stroke'].to_numpy().tolist()
    
    del df_tr['stroke']
    del df_de['stroke']
    del df_te['stroke']
    
    X_tr = df_tr.to_numpy().tolist()
    X_de = df_de.to_numpy().tolist()
    X_te = df_te.to_numpy().tolist()
    
    return X_tr, y_tr, X_de,y_de ,X_te,y_te           

def load_wines(self, format = "pandas", samples =3):
    import pandas as pd 
    import os 
    path = self.download()
    ptrain = os.path.join(path,'train.csv')
    pdev =  os.path.join(path,'dev.csv')
    ptest = os.path.join(path,'test.csv')
    
    df_tr = pd.read_csv(ptrain)
    df_de = pd.read_csv(pdev)
    df_te = pd.read_csv(ptest)
    
    y_tr = df_tr['price'].to_numpy().tolist()
    y_de = df_de['price'].to_numpy().tolist()
    y_te = df_te['price'].to_numpy().tolist()
    
    del df_tr['price']
    del df_de['price']
    del df_te['price']
    
    X_tr = df_tr.to_numpy().tolist()
    X_de = df_de.to_numpy().tolist()
    X_te = df_te.to_numpy().tolist()
    
    return X_tr, y_tr, X_de,y_de ,X_te,y_te

def load_women_clothing(self, format = "pandas", samples =3):
    import pandas as pd 
    import os 
    path = self.download()
    ptrain = os.path.join(path,'train.csv')
    pdev =  os.path.join(path,'dev.csv')
    ptest = os.path.join(path,'test.csv')
    
    
    df_tr = pd.read_csv(ptrain)
    df_de = pd.read_csv(pdev)
    df_te = pd.read_csv(ptest)
    
    y_tr = df_tr['Class Name'].to_numpy().tolist()
    y_de = df_de['Class Name'].to_numpy().tolist()
    y_te = df_te['Class Name'].to_numpy().tolist()
    
    del df_tr['Class Name']
    del df_de['Class Name']
    del df_te['Class Name']
    X_tr = df_tr.to_numpy().tolist()
    X_de = df_de.to_numpy().tolist()
    X_te = df_te.to_numpy().tolist()
    return X_tr, y_tr, X_de,y_de ,X_te,y_te

def load_project_kickstarter(self, format = "pandas", samples =3):
    import pandas as pd 
    import os 
    path = self.download()
    ptrain = os.path.join(path,'train.csv')
    pdev =  os.path.join(path,'dev.csv')
    ptest = os.path.join(path,'test.csv')

    
    # Read the values of the file in the datafrae
    df_tr = pd.read_csv(ptrain)
    df_te = pd.read_csv(ptest)
    df_de = pd.read_csv(pdev)

    y_tr = df_tr['final_status'].to_numpy().tolist()
    X_tr= df_tr.drop(['final_status','project_id'], axis=1).to_numpy().tolist()
    
    y_te = df_te['final_status'].to_numpy().tolist()
    X_te= df_te.drop(['final_status','project_id'], axis=1).to_numpy().tolist()
    
    y_de = df_de['final_status'].to_numpy().tolist()
    X_de= df_de.drop(['final_status','project_id'], axis=1).to_numpy().tolist()

    return X_tr,y_tr,X_de,y_de ,X_te,y_te

def load_jobs(self, format = "pandas", samples =3):
    import pandas as pd 
    import os 
    path = self.download()
    ptrain = os.path.join(path,'train.csv')
    ptest =  os.path.join(path,'dev.csv')
    pdev = os.path.join(path,'test.csv')
    
    df_tr = pd.read_csv(ptrain)
    df_de = pd.read_csv(pdev)
    df_te = pd.read_csv(ptest)
    
    y_tr = df_tr['fraudulent'].to_numpy().tolist()
    y_de = df_de['fraudulent'].to_numpy().tolist()
    y_te = df_te['fraudulent'].to_numpy().tolist()
    
    del df_tr['fraudulent']
    del df_de['fraudulent']
    del df_te['fraudulent']
    
    X_tr = df_tr.to_numpy().tolist()
    X_de = df_de.to_numpy().tolist()
    X_te = df_te.to_numpy().tolist()
    return X_tr, y_tr, X_de,y_de ,X_te,y_te

###################################### Image  #######################################
def load_fashion(self):
    import pandas as pd 
    import os 
    path = self.download()
    ptrain = os.path.join(path,'train.csv')
    ptest =  os.path.join(path,'dev.csv')
    pdev = os.path.join(path,'test.csv')
    
    df_tr = pd.read_csv(ptrain)
    df_de = pd.read_csv(pdev)
    df_te = pd.read_csv(ptest)
    
    y_tr = df_tr['label'].to_numpy().tolist()
    y_de = df_de['label'].to_numpy().tolist()
    y_te = df_te['label'].to_numpy().tolist()
    
    del df_tr['label']
    del df_de['label']
    del df_te['label']
    return df_tr, y_tr, df_de,y_de ,df_te,y_te 
# def load_paws_en(self, format = "pandas", samples =3):
#     '''
#     Return 
#     Train 
#     X_tr : list (list(string)) : Strings are sentences
#     y_tr : list(int)
#     Validation 
#     X_de : list (list(string)) : Strings are sentences
#     y_de : list(int)
#     Test
#     X_te : list (list(strings)) Strings are sentences
#     y_te : list(int)
    
#     '''
    
#     import pandas as pd 
#     import os 
#     path = self.download()

#     train = os.path.join(path,'train1.tsv')
#     dev =  os.path.join(path,'dev.tsv')
#     test =  os.path.join(path,'test.tsv')
#     dftr = pd.read_table(train)
#     dfte = pd.read_table(test)
#     dfde = pd.read_table(dev)
    
#     dummy_tr = dftr.filter(regex='(sentence1|sentence2)') 
#     y_tr = dftr['label'].to_numpy().tolist() 
#     X_tr = dummy_tr.to_numpy().tolist()

#     dummy_de = dfde.filter(regex='(sentence1|sentence2)') 
#     y_de = dfde['label'].to_numpy().tolist()
#     X_de = dummy_de.to_numpy().tolist()
    
#     dummy_te = dfte.filter(regex='(sentence1|sentence2)') 
#     y_te = dfte['label'].to_numpy().tolist()
#     X_te = dummy_te.to_numpy().tolist()
#     return X_tr,y_tr, X_de, y_de, X_te,y_te   

# def load_wnli(self, format = "pandas", samples =3):
#     import pandas as pd 
#     import os 
#     import numpy as np
#     path = self.download()
    

#     ptrain = os.path.join(path,'train.txt')
#     pdev = os.path.join(path,'dev.txt')
#     ptest = os.path.join(path,'test.txt')
    
#     dftr = pd.read_csv(ptrain,sep="\t")
    
#     dfde = pd.read_csv(pdev,sep="\t")
    
#     dfte = pd.read_csv(ptest,sep="\t")
#     # ptrain = os.path.join(path,'train.tsv')
#     # pdev =  os.path.join(path,'dev.tsv')
#     # ptest =  os.path.join(path,'test.tsv')

#     # dftr = pd.read_table(ptrain)
#     # dfte = pd.read_table(ptest)
#     # dfde = pd.read_table(pdev)

#     dummy_tr = dftr.filter(regex='(sentence1|sentence2)') 
#     y_tr = list(dftr['label'])
#     X_tr = dummy_tr.to_numpy().tolist()

#     dummy_de = dfde.filter(regex='(sentence1|sentence2)') 
#     y_de = list(dfde['label'])
#     X_de = dummy_de.to_numpy().tolist()
    
#     dummy_te = dfte.filter(regex='(sentence1|sentence2)') 
#     y_te = list(dfte['label'])

#     X_te = list(dummy_te.iloc[:, 0:2])
#     return X_tr,y_tr, X_de, y_de, X_te,y_te  