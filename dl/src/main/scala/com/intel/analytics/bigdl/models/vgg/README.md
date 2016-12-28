# Vgg Model on Cifar10
In this example, we will train vgg model on Cifar10 dataset. The original model is from [this
article](http://torch.ch/blog/2015/07/30/cifar.html).

## Prepare Cifar10 Dataset
You can download Cifar10 dataset from [this webpage](https://www.cs.toronto.edu/~kriz/cifar.html)
.

## Train Model
Example command
```
./dist/bin/bigdl.sh -- java -cp dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar com.intel.analytics.bigdl.models.vgg.Train --core 28 --node 1 --env local -f Cifar-10/ --checkpoint .
```

## Train Model on Spark
Local mode command
```
./dist/bin/bigdl.sh -- spark-submit --master local[28] --class com.intel.analytics.bigdl.models.vgg.Train dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar  --core 28 --node 1 --env spark -f Cifar-10/ -b 112 --checkpoint .
```

Cluster mode command
```

./dist/bin/bigdl.sh -- spark-submit --class com.intel.analytics.bigdl.models.vgg.Train dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar  --core 28 --node 4 --env spark -f Cifar-10/ -b 448 --checkpoint .
```

## Test Model
Example command
```
./dist/bin/bigdl.sh -- java -cp dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar com.intel.analytics.bigdl.models.vgg.Test -f Cifar-10/val --model model.113 --nodeNumber 1 --core 28 --env local
```

Spark Local mode command
```
./dist/bin/bigdl.sh -- spark-submit --master local[28] --class com.intel.analytics.bigdl.models.vgg.Test dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-all-in-one.jar -f Cifar-10/val --model model.113 --nodeNumber 1 --core 28 --env spark -b 112
```

Spark cluster mode command
```
./dist/bin/bigdl.sh -- spark-submit --class com.intel.analytics.bigdl.models.vgg.Test dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-all-in-one.jar -f Cifar-10/val --model model.113 --nodeNumber 4 --core 28 --env spark -b 448
```