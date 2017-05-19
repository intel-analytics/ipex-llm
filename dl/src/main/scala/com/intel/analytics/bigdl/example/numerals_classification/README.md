# Numerals Classification

This example is to do basic logistic regression with ReLU in deep network. While BigDL is made for deep networks it can likewise represent "shallow" models like logistic regression for classification.

We'll do simple logistic regression on synthetic data feed vectors to BigDL.

## Prepare Data
Synthesize a dataset of 10,000 4-vectors for binary classification with 2 informative features and 2 noise features.

Before generate dataset, please install python sklearn package. 

i.e. In Ubuntu, run "sudo apt-get update; sudo apt-get install python-sklearn"

Use script data_gen.py to generate data in target directory:

usage: python data_gen.py [-h] dir_name

Example: python data_gen.py ~/data

## Get the JAR
You can download one from [here]() or build one by refer to the
[Build Page](https://github.com/intel-analytics/BigDL/wiki/Build-Page) from the source code

## Train the Model
Example command                     
```
dist/bin/bigdl.sh -- java -cp dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar com.intel.analytics.bigdl.example.numerals_classification.Train -f ~/data --core 4 --node 1 --env local --checkpoint ~/model
```
### Use Apache Spark
Local mode, example command
```
./dist/bin/bigdl.sh -- spark-submit --master local[4] --driver-class-path dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar --class com.intel.analytics.bigdl.example.numerals_classification.Train dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar -f ~/data --core 4 --node 1 -b 12 --checkpoint ~/model --env spark
```
Cluster mode, example command
```
./dist/bin/bigdl.sh -- spark-submit --driver-class-path dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar --class com.intel.analytics.bigdl.example.numerals_classification.Train dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar -f ~/data --core 4 --node 4 --env spark -b 48
```
## Test Model
The above commands will cache the model in specified path(--checkpoint). Run this command will
use the model to do a validation.

Example command
```
dist/bin/bigdl.sh -- java -cp dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar com.intel.analytics.bigdl.example.numerals_classification.Test -f ~/data --core 4 -n 1 --env local --model ~/model/model.9001
```
Spark local mode, example command
```
./dist/bin/bigdl.sh -- spark-submit --master local[4] --class com.intel.analytics.bigdl.example.numerals_classification.Test dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar -f ~/data --model ~/model/model.7501 --node 1 --core 28 --env spark -b 448
```
Spark cluster mode, example command
```
./dist/bin/bigdl.sh -- spark-submit --class com.intel.analytics.bigdl.example.numerals_classification.Test dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar -f ~/data --model ~/model/model.12001 --node 4 --core 28 --env spark -b 896
```