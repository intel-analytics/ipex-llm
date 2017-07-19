

## **Run as a local Java/Scala program**
You can try BigDL program, e.g., the [lenet](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/example/lenetLocal) training, testing and prediction as a local Java/Scala program. 

To run the BigDL model as a local Java/Scala program, user needs to set Java property `bigdl.localMode` to `true`. If user wants to specify how many cores to be used for training/testing/prediction, he needs to set Java property `bigdl.coreNumber` to the core number. User can either call `System.setProperty("bigdl.localMode", "true")` and `System.setProperty("bigdl.coreNumber", core_number)` in the Java/Scala code, or pass -Dbigdl.localMode=true and -Dbigdl.coreNumber=core_number when runing the program.

1. First, you can download the MNIST Data from [here](http://yann.lecun.com/exdb/mnist/). Unzip all the files and put them in one folder(e.g. mnist).

2. Run below command to train lenet as local Java/Scala program:
```bash
java -cp spark/dl/target/bigdl-VERSION-jar-with-dependencies-and-spark.jar \
com.intel.analytics.bigdl.example.lenetLocal.Train \
-f path_to_mnist_folder \
-c core_number \
-b batch_size \
--checkpoint ./model
```
In the above commands
* -f: where you put your MNIST data
* -c: The core number on local machine used for this training. The default value is physical cores number. Get it through Runtime.getRuntime().availableProcessors() / 2
* -b: The mini-batch size. It is expected that the mini-batch size is a multiple of core_number
* --checkpoint: Where you cache the model/train_state snapshot. You should input a folder and
make sure the folder is created when you run this example. The model snapshot will be named as
model.#iteration_number, and train state will be named as state.#iteration_number. Note that if
there are some files already exist in the folder, the old file will not be overwrite for the
safety of your model files.

3. The above commands will cache the model in specified path(--checkpoint). Run this command will
   use the trained model to do a validation.
```
java -cp spark/dl/target/bigdl-VERSION-jar-with-dependencies-and-spark.jar \
com.intel.analytics.bigdl.example.lenetLocal.Test \
-f path_to_mnist_folder \
--model ./model/model.iteration \
-c core_number \
-b batch_size
```
In the above command
* -f: where you put your MNIST data
* --model: the model snapshot file
* -c: The core number on local machine used for this testing. The default value is physical cores number. Get it through Runtime.getRuntime().availableProcessors() / 2
* -b: The mini-batch size. It is expected that the mini-batch size is a multiple of core_number   
   
4. Run below command to predict with trained model:
```
java -cp spark/dl/target/bigdl-VERSION-jar-with-dependencies-and-spark.jar \
com.intel.analytics.bigdl.example.lenetLocal.Predict \
-f path_to_mnist_folder \
-c core_number \
--model ./model/model.iteration
```
In the above command
* -f: where you put your MNIST data
* -c: The core number on local machine used for this prediction. The default value is physical cores number. Get it through Runtime.getRuntime().availableProcessors() / 2
* --model: the model snapshot file
