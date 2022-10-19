def load_wnli(self):
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

    dummy_tr = dftr.filter(regex='(sentence1|sentence2)') 
    y_tr = list(dftr['label'])
    X_tr = dummy_tr.to_numpy().tolist()

    dummy_de = dfde.filter(regex='(sentence1|sentence2)') 
    y_de = list(dfde['label'])
    X_de = dummy_de.to_numpy().tolist()
    
    dummy_te = dfte.filter(regex='(sentence1|sentence2)') 
    y_te = list(dfte['label'])

    X_te = list(dummy_te.iloc[:, 0:2])
    return X_tr,y_tr, X_de, y_de, X_te,y_te  
