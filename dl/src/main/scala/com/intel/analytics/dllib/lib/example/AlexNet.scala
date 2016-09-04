package com.intel.analytics.dllib.lib.example

import java.util

import com.intel.analytics.dllib.lib.nn._
import com.intel.analytics.dllib.lib.optim.SGD
import com.intel.analytics.dllib.lib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.dllib.lib.tensor._
import com.intel.analytics.dllib.lib.tensor.TensorNumericMath.TensorNumeric._

import scala.reflect.ClassTag

/*
 * Compare with torch AlexNet
 */
object AlexNet {

  def main(args : Array[String]) : Unit = {
    require(args.length >= 1)
    args(0) match {
      case "perf" => performance(args(1).toInt, args(2).toInt, args(3))
      case _ => throw new IllegalArgumentException
    }
  }

  def performance(batchSize : Int, iter : Int, dtype : String) : Unit = {
    var forwardTime = 0L
    var backwardTime = 0L
    val times = dtype.toLowerCase match {
      case "float" =>
        val input = torch.Tensor[Float](batchSize, 3, 224, 224).fill(0.5f)
        val model = getModelCaffeOWT[Float](1000)
        val (parm, grad) = model.getParameters()
        println(model)
        println(parm.nElement())
        val criterion = new ClassNLLCriterion[Float]()
        val labelData = new Array[Float](batchSize)
        util.Arrays.fill(labelData, 10)
        val labels = torch.Tensor[Float](torch.storage(labelData))
        var i = 0
        println("warm up")
        while(i < 5) {
          val output = model.forward(input)
          val loss = criterion.forward(output, labels)
          val gradOutput = criterion.backward(output, labels)
          model.backward(input, gradOutput)
          i += 1
        }
        println("warm up done")
        model.resetTimes()
        i = 0
        while(i < iter) {
          var start = System.nanoTime()
          val output = model.forward(input)
          criterion.forward(output, labels)
          forwardTime += System.nanoTime() - start
          start = System.nanoTime()
          val gradOutput = criterion.backward(output, labels)
          model.backward(input, gradOutput)
          backwardTime += System.nanoTime() - start
          i += 1
        }
        model.getTimes()
      case "double"=>
        val input = torch.Tensor[Double](batchSize, 3, 224, 224).fill(0.5)
        val model = getModelCaffeOWT[Double](1000)
        val (parm, grad) = model.getParameters()
        println(model)
        println(parm.nElement())
        val criterion = new ClassNLLCriterion[Double]()
        val labelData = new Array[Double](batchSize)
        util.Arrays.fill(labelData, 10)
        val labels = torch.Tensor[Double](torch.storage(labelData))
        var i = 0
        println("warm up")
        while(i < 5) {
          val output = model.forward(input)
          criterion.forward(output, labels)
          val gradOutput = criterion.backward(output, labels)
          model.backward(input, gradOutput)
          i += 1
        }
        println("warm up done")
        model.resetTimes()
        i = 0
        while(i < iter) {
          var start = System.nanoTime()
          val output = model.forward(input)
          val loss = criterion.forward(output, labels)
          forwardTime += System.nanoTime() - start
          start = System.nanoTime()
          val gradOutput = criterion.backward(output, labels)
          model.backward(input, gradOutput)
          backwardTime += System.nanoTime() - start
          i += 1
        }
        model.getTimes()
      case _ => throw new IllegalArgumentException
    }
    println(s"forward time is ${forwardTime / iter / 1e6}ms")
    println(s"backward time is ${backwardTime / iter / 1e6}ms")
    println(s"total time is ${(forwardTime + backwardTime) / iter / 1e6}ms")

    var n = 0
    println(times.map(t => ({n += 1; s"${t._1}-$n"}, (t._2 + t._3) / 1e9 / iter, t._2 / 1e9 / iter, t._3 / 1e9 / iter))
      .sortWith(_._2 > _._2).mkString("\n"))
    n = 0
    println(times.filter(_._1.isInstanceOf[SpatialConvolution[_]])
      .map(t => ({n += 1; s"${t._1}-$n"}, t._1.asInstanceOf[SpatialConvolution[_]]))
      .map(t => (t._1, t._2.getIm2ColTime() / 1e9 / iter, t._2.getCol2ImgTime() / 1e9 / iter))
      .mkString("\n"))

    System.exit(0)
  }

  //this is AlexNet that was presented in the One Weird Trick paper. http://arxiv.org/abs/1404.5997
  def getModel[T : ClassTag](classNum : Int)(implicit ev: TensorNumeric[T]): Module[T] = {
    val feature = new Sequential[T]
    feature.add(new SpatialConvolution[T](3, 64, 11, 11, 4, 4, 2, 2))
    feature.add(new ReLU[T](true))
    feature.add(new SpatialMaxPooling[T](3, 3, 2, 2))
    feature.add(new SpatialConvolution[T](64, 192, 5, 5, 1, 1, 2, 2))
    feature.add(new ReLU[T](true))
    feature.add(new SpatialMaxPooling[T](3, 3, 2, 2))
    feature.add(new SpatialConvolution[T](192, 384, 3, 3, 1, 1, 1, 1))
    feature.add(new ReLU[T](true))
    feature.add(new SpatialConvolution[T](384, 256, 3, 3, 1, 1, 1, 1))
    feature.add(new ReLU[T](true))
    feature.add(new SpatialConvolution[T](256, 256, 3, 3, 1, 1, 1, 1))
    feature.add(new ReLU[T](true))
    feature.add(new SpatialMaxPooling[T](3, 3, 2, 2))



    val classifier = new Sequential[T]
    classifier.add(new View[T](256 * 6 * 6))
    classifier.add(new Dropout[T](0.5))
    classifier.add(new Linear[T](256 * 6 * 6, 4096))
    classifier.add(new Threshold[T](0, 1e-6))
    classifier.add(new Dropout[T](0.5))
    classifier.add(new Linear[T](4096, 4096))
    classifier.add(new Threshold[T](0, 1e-6))
    classifier.add(new Linear[T](4096, classNum))
    classifier.add(new LogSoftMax[T])


    val model = new Sequential[T]
    model.add(feature).add(classifier)

    model
  }

  def getModelCaffeOWT[T : ClassTag](classNum : Int)(implicit ev: TensorNumeric[T]): Module[T] = {
    val feature = new Sequential[T]
    feature.add(new SpatialConvolution[T](3, 64, 11, 11, 4, 4, 2, 2))
    feature.add(new ReLU[T](true))
    feature.add(new SpatialMaxPooling[T](3, 3, 2, 2))
    feature.add(new SpatialConvolution[T](64, 192, 5, 5, 1, 1, 2, 2))
    feature.add(new ReLU[T](true))
    feature.add(new SpatialMaxPooling[T](3, 3, 2, 2))
    feature.add(new SpatialConvolution[T](192, 384, 3, 3, 1, 1, 1, 1))
    feature.add(new ReLU[T](true))
    feature.add(new SpatialConvolution[T](384, 256, 3, 3, 1, 1, 1, 1))
    feature.add(new ReLU[T](true))
    feature.add(new SpatialConvolution[T](256, 256, 3, 3, 1, 1, 1, 1))
    feature.add(new ReLU[T](true))
    feature.add(new SpatialMaxPooling[T](3, 3, 2, 2))



    val classifier = new Sequential[T]
    classifier.add(new View[T](256 * 6 * 6))
    classifier.add(new Linear[T](256 * 6 * 6, 4096))
    classifier.add(new Linear[T](4096, 4096))
    classifier.add(new Linear[T](4096, classNum))
    classifier.add(new LogSoftMax[T])


    val model = new Sequential[T]
    model.add(feature).add(classifier)

    model
  }
}
