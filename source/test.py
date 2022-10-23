from benchmark import AutoMLBench

AutoMLBench.init()
train,x= AutoMLBench.load_dataset("paws-x-es",in_xy=False,samples=2)
print(train)
#train,y_train, test ,y_test = AutoMLBench.load_dataset("sst-en", in_xy = True, samples = 2)

#train = AutoMLBench.load_dataset("sst-en", in_xy = False, samples = 1)

#print(train)
