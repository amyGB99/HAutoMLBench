def load_paws_en(self):
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
    dev =  os.path.join(path,'dev.txt')
    test =  os.path.join(path,'test.txt')
    dftr = pd.read_csv(train,sep='\t')
    dfte = pd.read_csv(test,sep='\t')
    dfde = pd.read_csv(dev,sep='\t')
    
    dummy_tr = dftr.filter(regex='(sentence1|sentence2)') 
    y_tr = list(dftr['label'])
    X_tr = dummy_tr.to_numpy().tolist()

    dummy_de = dfde.filter(regex='(sentence1|sentence2)') 
    y_de = list(dfde['label'])
    X_de = dummy_de.to_numpy().tolist()
    
    dummy_te = dfte.filter(regex='(sentence1|sentence2)') 
    y_te = list(dfte['label'])
    X_te = dummy_te.to_numpy().tolist()
    return X_tr,y_tr, X_de, y_de, X_te,y_te   
