from benchmark import AutoMLBench
# from autogluon.multimodal import MultiModalPredictor
# from autogluon.text import TextPredictor
import pandas as pd 
AutoMLBench.init()

names  =AutoMLBench.names
print(names)
# for dataset in names:
#     try:
#         print(dataset)
#         if dataset =='meddocan':
#             continue
#         #train,test = AutoMLBench.load_dataset(dataset,format='pandas',in_xy=False,samples=2)
#         all = AutoMLBench.load_dataset(dataset,format='pandas',in_xy=False,samples=1)
#         print(all.isnull().sum())
#         print(all['label'].unique())
#         #print(all)
        
#         #print(train)
#     except:
#         if dataset =='women-clothing': 
#             print(all['Class Name'].unique())
#             continue
#         if dataset =='stroke-prediction': 
#             print(all['stroke'].unique())
#             continue
#         if dataset =='fraudulent-jobs': 
#             print(all['fraudulent'].unique())
#             continue
#         if dataset =='spanish-wine': 
#             print(all['price'].unique())
#             continue
#         if dataset =='project-kickstarter': 
#             print(all['final_status'].unique())
#             continue
#         if dataset =='inferes': 
#             print(all['Label'].unique())
#             continue
#         if dataset =='predict-salary': 
#             print(all['salary'].unique())
#             continue
#         if dataset =='twiteer-human-bots': 
#             print(all['account_type'].unique())
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