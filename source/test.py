from benchmark import AutoMLBench
# from autogluon.multimodal import MultiModalPredictor
# from autogluon.text import TextPredictor
import pandas as pd 
AutoMLBench.init()
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
train, test,  = AutoMLBench.load_dataset("google-guest",format='pandas',in_xy=False,samples=2)


print(train)
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