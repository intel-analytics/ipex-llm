# VGG Model on CIFAR-10
This example demonstrates how to use BigDL to train and test a [VGG-like](http://torch.ch/blog/2015/07/30/cifar.html) network on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) data.

## Prepare CIFAR-10 Dataset
You can download CIFAR-10 dataset from [this webpage](https://www.cs.toronto.edu/~kriz/cifar.html) (remember to choose the binary version).

## Train Model
Example command for running as a local Java program
```
./dist/bin/bigdl.sh -- java -cp dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar com.intel.analytics.bigdl.models.vgg.Train --core 28 --node 1 --env local -f Cifar-10/ --checkpoint .
```

Example command for running in Spark local mode
```
./dist/bin/bigdl.sh -- spark-submit --master local[28] --class com.intel.analytics.bigdl.models.vgg.Train dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar  --core 28 --node 1 --env spark -f Cifar-10/ -b 112 --checkpoint .
```

Example command for running in Spark cluster mode
```

./dist/bin/bigdl.sh -- spark-submit --class com.intel.analytics.bigdl.models.vgg.Train dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar  --core 28 --node 4 --env spark -f Cifar-10/ -b 448 --checkpoint .
```

## Test Model
Example command for running as a local Java program
```
./dist/bin/bigdl.sh -- java -cp dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar com.intel.analytics.bigdl.models.vgg.Test -f Cifar-10/val --model model.113 --nodeNumber 1 --core 28 --env local
```

Example command for running in Spark local mode
```
./dist/bin/bigdl.sh -- spark-submit --master local[28] --class com.intel.analytics.bigdl.models.vgg.Test dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-all-in-one.jar -f Cifar-10/val --model model.113 --nodeNumber 1 --core 28 --env spark -b 112
```

Example command for running in Spark cluster mode
```
./dist/bin/bigdl.sh -- spark-submit --class com.intel.analytics.bigdl.models.vgg.Test dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-all-in-one.jar -f Cifar-10/val --model model.113 --nodeNumber 4 --core 28 --env spark -b 448
```
