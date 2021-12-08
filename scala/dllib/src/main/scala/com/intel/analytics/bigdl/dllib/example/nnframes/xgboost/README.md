# XGBoostClassifier Train Example

## Environment
- spark 2.4.6
- BigDL 2.0 
## Data Prepare

### Build BigDL/scala
run :
```
bash /path/to/BigDL/scala/make-dist.sh
```

You will get jars include `bigdl-dllib-spark_2.4.6-0.14.0-SNAPSHOT-jar-with-dependencies.jar`

### Get iris.csv

run python command:
```
from sklearn.datasets import load_iris
import numpy as np
import pandas

X, y = load_iris(return_X_y=True)
y = y.astype(np.int)
df = pandas.DataFrame(data=X, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
class_id_to_name = {0:'Iris-setosa', 1:'Iris-versicolor', 2:'Iris-virginica'}
df['class'] = np.vectorize(class_id_to_name.get)(y)
df.to_csv('./iris.csv', float_format='%.1f', header=False, index=False)
```

You will get file `iris.csv` in current path.

## Run:
command:
```
spark-submit \
  --master local[2] \
  --conf spark.task.cpus=2  \
  --class com.intel.analytics.bigdl.dllib.examples.nnframes.xgboost.xgbClassifierTrainingExample \
  /path/to/BigDL/scala/dllib/target/bigdl-dllib-spark_2.4.6-0.14.0-SNAPSHOT-jar-with-dependencies.jar \
  /path/to/iris.csv 2 100 /path/to/model/saved
```

You will get output like:
```
[INFO] [12/08/2021 16:05:41.989] [RabitTracker-akka.actor.default-dispatcher-22] [akka://RabitTracker/user/Handler] [591298]    train-merror:0.000000   eval1-merror:0.000000     eval2-merror:0.125000
```
And tree of folder `/path/to/model/saved` :
```
.
├── data
│   └── XGBoostClassificationModel
└── metadata
    ├── part-00000
    └── _SUCCESS
```
parameters:
- path_to_iris.csv : String
- num_threads : Int
- num_round : Int 
- path_to_model_saved : String

note: make sure num_threads is larger than spark.task.cpus.
