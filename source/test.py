from benchmark import AutoMLBench

AutoMLBench.init()
x = AutoMLBench.load_dataset("wnli-es",in_xy=False,samples=1)
print(x)