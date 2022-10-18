def load_sst_en(self):
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
