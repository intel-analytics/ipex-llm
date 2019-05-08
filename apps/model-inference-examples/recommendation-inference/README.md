# Simple Service Sample of recommendation

## Summary
This is a simple service exmaple of Neural collaborative filtering model in Scala and Java.You can plug it into yuor web service or local program. 

To run this example, please follow the steps below.

### Run Application

Step 1. Follow the [example in the zoo](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/recommendation/README.md) to train a NeuralCFModel. You want to modify the [example](https://github.com/intel-analytics/analytics-zoo/blob/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/recommendation/NeuralCFexample.scala) and save the trained model into "./models/ncf.bigdl".
   
Step 2. Run the example `SimpleDriver.java` or `SimpleScalaDriver.scala`, simply running it can get the prediction result in the terminal.

