from benchmark import AutoMLBench
# from autogluon.multimodal import MultiModalPredictor
# from autogluon.text import TextPredictor
import pandas as pd 
AutoMLBench.init()

names  =AutoMLBench.names
print(names)
for dataset in names:
    if dataset == "wikiann-es" or dataset =='meddocan' or dataset =='wikineural-es' or dataset =='wikineural-en':
        continue
    #all = AutoMLBench.load_dataset(dataset,format='pandas',in_xy=False,samples=1)
    #print(all)
    train,test = AutoMLBench.load_dataset(dataset,format='pandas',in_xy=False,samples=2)
    print(test)
#     try:
#         print(dataset)
#         #train,test = AutoMLBench.load_dataset(dataset,format='pandas',in_xy=False,samples=2)
#         print(len(train.axes[0]))
#         print(len(test.axes[0]))
#         #print(train['label'].values_count())
        
#         #print(all.isnull().sum())
#         #print(all['label'].unique())
#         #print(all['label'].unique())
        
# #         #print(all)
        
# #         #print(train)
#     except:
#         if dataset =='women-clothing': 
#             #print(all['Class Name'].unique())
#             print(len(train.axis[0]))
#             print(len(test.axis[0]))
#             continue
#         if dataset =='stroke-prediction': 
#             #print(all['stroke'].unique())
#             print(len(train.axis[0]))
#             print(len(test.axis[0]))
#             continue
#         if dataset =='fraudulent-jobs': 
#            # print(all['fraudulent'].unique())
#             print(len(train.axis[0]))
#             print(len(test.axis[0]))
#             continue
#         if dataset =='spanish-wine': 
#             #print(all['price'].unique())
#             print(len(train.axis[0]))
#             print(len(test.axis[0]))
#             continue
#         if dataset =='project-kickstarter': 
#             #print(all['final_status'].unique())
#             print(len(train.axis[0]))
#             print(len(test.axis[0]))
#             continue
#         if dataset =='inferes': 
#             #print(all['Label'].unique())
#             print(len(train.axis[0]))
#             print(len(test.axis[0]))
#             continue
#         if dataset =='predict-salary': 
#             #print(all['salary'].unique())
#             print(len(train.axis[0]))
#             print(len(test.axis[0]))
#             continue
#         if dataset =='twitter-human-bots': 
#             #print(all['account_type'].unique())
#             print(len(train.axis[0]))
#             print(len(test.axis[0]))
#             continue
#         if dataset =='google-guest': 
#             #print(all['question_well_written'].unique())
#             print(len(train.axis[0]))
#             print(len(test.axis[0]))
#             continue
#         if dataset =='language-identification': 
#             #print(all['labels'].unique())
#             print(len(train.axis[0]))
#             print(len(test.axis[0]))
#             continue
            
#         print(f'Error{dataset}')

#train,y_train, test ,y_test= AutoMLBench.load_dataset("stsb-en",format='list',in_xy=True,samples=2)
#train,y_train, test ,y_test= AutoMLBench.load_dataset("stsb-es",in_xy=True,samples=2)
#train,y_train, test ,y_test = AutoMLBench.load_dataset("predict-salary",format='list',in_xy=True,samples=2)
#train,test = AutoMLBench.load_dataset("women-clothing",format='pandas',in_xy=False,samples=2)
#AutoMLBench.load_dataset("fraudulent-jobs",format='pandas',in_xy=False,samples=2)
#train,test = AutoMLBench.load_dataset("inferes",format='pandas',in_xy=False,samples=2)
#train,ytr, test,yte = AutoMLBench.load_dataset("spanish-wine",format='list',in_xy=True,samples=2)
#X_train, y_train, X_test, y_test = AutoMLBench.load_dataset("meddocan",format='pandas',in_xy=False,samples=2)
#train, test,  = AutoMLBench.load_dataset("vaccine-en",format='pandas',in_xy=False,samples=2)
#train, test,  = AutoMLBench.load_dataset("language-identification",format='pandas',in_xy=False,samples=2)
#train, test,  = AutoMLBench.load_dataset("twiter-human-bots",format='pandas',in_xy=False,samples=2)
#train,test = AutoMLBench.load_dataset("sentiment-lexicons-es",format='pandas',in_xy=False,samples=2)
#all,y = AutoMLBench.load_dataset("sentiment-lexicons-es",format='pandas',in_xy=True,samples=1)


#train,y_train, test ,y_test = AutoMLBench.load_dataset("sst-en", in_xy = True, samples = 2)

#train = AutoMLBench.load_dataset("sst-en", in_xy = False, samples = 1)

#print(train)
# train_data,test_data = AutoMLBench.load_dataset("paws-x-en",format='pandas',in_xy=False,samples=2)
# print("fit")
# time_limit = 10*60
# predictor = TextPredictor(label='label').fit(train_data,time_limit=time_limit)
# print("predict")
# predictions = predictor.predict(test_data)
# print("evaluate")
# score = predictor.evaluate(test_data)
# print(score)