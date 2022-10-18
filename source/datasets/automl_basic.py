# AutoGOAL Example: basic usage of the AutoML class

from autogoal.datasets import cars
from autogoal.kb import MatrixContinuousDense, Supervised, VectorCategorical
from autogoal.ml import AutoML
from sklearn.model_selection import train_test_split

# Load dataset
X, y = cars.load()

# Instantiate AutoML and define input/output types
automl = AutoML(
    input=(MatrixContinuousDense, Supervised[VectorCategorical]),
    output=VectorCategorical,
)

# Run the pipeline search process
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#print(X_train)
#print(y_train)
print("init train")
automl.fit(X_train, y_train)

# Report the best pipeline
print("best_pipeline")
print(automl.best_pipeline_)
print("best_score")
print(automl.best_score_)
print("init score")
score = automl.score(X_test, y_test)
print("final score")

# Let's see what we got!

print(score)