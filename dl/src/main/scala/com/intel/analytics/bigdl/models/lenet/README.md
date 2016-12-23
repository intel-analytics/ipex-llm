# LeNet5 Model on MNIST

LeNet5 is a classical CNN model used in digital number classification. For detail information,
please refer to <http://yann.lecun.com/exdb/lenet/>.

## Prepare MNIST Data
You can download the MNIST Data from [here](http://yann.lecun.com/exdb/mnist/). Unzip all the
files and put them in one folder(e.g. mnist).

There're four files. **train-images-idx3-ubyte** contains train images,
**train-labels-idx1-ubyte** is train label file, **t10k-images-idx3-ubyte** has validation images
 and **t10k-labels-idx1-ubyte** contains validation labels. For more detail, please refer to the
 download page.

## Get the JAR
You can download one from [here]() or build one by refer to the
[Build Page](https://github.com/intel-analytics/BigDL/wiki/Build-Page) from the source code

## Train the Model
Example command
```
dist/bin/bigdl.sh -- java -cp dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar com.intel.analytics.bigdl.models.lenet.Train -f ~/mnist/ --core 4 --node 1 --env local --checkpoint ~/model
```
### Use Apache Spark
Local mode, example command
```
./dist/bin/bigdl.sh -- spark-submit --master local[4] --driver-class-path dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar --class com.intel.analytics.bigdl.models.lenet.Train dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar -f ~/mnist/ --core 4 --node 1 --env spark
```
Cluster mode, example command
```
./dist/bin/bigdl.sh -- spark-submit --driver-class-path dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar --class com.intel.analytics.bigdl.models.lenet.Train dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar -f ~/mnist/ --core 4 --node 4 --env spark -b 48
```
## Test Model
The above commands will cache the model in specified path(--checkpoint). Run this command will
use the model to do a validation.

Example command
```
dist/bin/bigdl.sh -- java -cp dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar com.intel.analytics.bigdl.models.lenet.Test -f ~/mnist/ --core 4 -n 1 --env local --model ~/model/model.90001
```
Spark local mode, example command
```
./dist/bin/bigdl.sh -- spark-submit --master local[4] --class com.intel.analytics.bigdl.models.lenet.Test dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar -f mnist/ --model model.12001 --nodeNumber 1 --core 28 --env spark -b 448
```
Spark cluster mode, example command
```
./dist/bin/bigdl.sh -- spark-submit --class com.intel.analytics.bigdl.models.lenet.Test dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar -f mnist/ --model model.12001 --nodeNumber 4 --core 28 --env spark -b 896
```