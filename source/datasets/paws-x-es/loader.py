def load_two_sentences(self,format = "pandas",complet = False ,samples =  2):
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
    if complet == True:
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
