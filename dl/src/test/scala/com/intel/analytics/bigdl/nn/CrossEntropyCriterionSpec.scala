package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.optim.SGD
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

import scala.math._


/**
  * Created by ywan on 16-9-21.
  */

class CrossEntropyCriterionSpec extends FlatSpec with Matchers {

  "CrossEntropyCriterion " should "return return right output and gradInput" in {
    val criterion = new CrossEntropyCriterion[Double]()

    val input = Tensor[Double](3, 3)
    input(Array(1, 1)) = 0.33655226649716
    input(Array(1, 2)) = 0.77367000770755
    input(Array(1, 3)) = 0.031494265655056
    input(Array(2, 1)) = 0.11129087698646
    input(Array(2, 2)) = 0.14688249188475
    input(Array(2, 3)) = 0.49454387230799
    input(Array(3, 1)) = 0.45682632108219
    input(Array(3, 2)) = 0.85653987620026
    input(Array(3, 3)) = 0.42569971177727

    val target = Tensor[Double](3)
    target(Array(1)) = 1
    target(Array(2)) = 2
    target(Array(3)) = 3
    //println("input.dim() = " + input.dim())

    val expectedOutput = 1.2267281042702334

    val loss = criterion.forward(input, target)
    loss should be(expectedOutput +- 1e-8)

    val expectedGrad = Tensor[Double](3, 3)
    expectedGrad(Array(1, 1)) = -0.23187185
    expectedGrad(Array(1, 2)) = 0.15708656
    expectedGrad(Array(1, 3)) = 0.07478529
    expectedGrad(Array(2, 1)) = 0.09514888
    expectedGrad(Array(2, 2)) = -0.23473696
    expectedGrad(Array(2, 3)) = 0.13958808
    expectedGrad(Array(3, 1)) = 0.09631823
    expectedGrad(Array(3, 2)) = 0.14364876
    expectedGrad(Array(3, 3)) = -0.23996699
    val gradInput = criterion.backward(input, target)

    expectedGrad.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })

  }

  "Multi-Classification LR " should "converge correctly" in {
    def specifiedModel(): Module[Tensor[Double], Tensor[Double], Double] = {
      val model = new Sequential[Tensor[Double], Tensor[Double], Double]()
      val linear = new Linear[Double](3, 3)
      linear.weight(Array(1, 1)) = -0.3
      linear.weight(Array(1, 2)) =  1.6
      linear.weight(Array(1, 3)) =  -0.3

      linear.weight(Array(2, 1)) = 1.4
      linear.weight(Array(2, 2)) = -0.4
      linear.weight(Array(2, 3)) = -0.6

      linear.weight(Array(3, 1)) =  -0.3
      linear.weight(Array(3, 2)) = -0.2
      linear.weight(Array(3, 3)) =  1.9

      linear.bias(Array(1)) = 0.0
      linear.bias(Array(2)) = 0.0
      linear.bias(Array(3)) = 0.0
      //linear.bias(Array(3)) = 0.02

      model.add(linear)
      //model.add(new LogSoftMax[Double]())
      model.add(new Sigmoid())
      model
    }

    def getTrainModel(): Module[Tensor[Double], Tensor[Double], Double] = {
      val model = new Sequential[Tensor[Double], Tensor[Double], Double]()
      model.add(new Linear[Double](3, 3))
      //model.add(new LogSoftMax[Double]())
      model.add(new Sigmoid[Double]())
      model
    }

    def feval(grad: Tensor[Double], module: Module[Tensor[Double], Tensor[Double], Double], criterion: TensorCriterion[Double],
              input: Tensor[Double], target: Tensor[Double])(weights: Tensor[Double])
    : (Double, Tensor[Double]) = {
      //println("enter feval")
      module.training()
      grad.zero()
      //val trainSize = input.size(1)
      //println("trainSize = " + trainSize)
      val output = module.forward(input) //.resize(Array(trainSize, 3))
      /*println("size2 = " + output.size(2))
      println("inputs:")
      for (i <- 1 to 10) {
        println(output(i))
      }
      println("targets:")
      for (i <- 1 to 10) {
        println(target(i))
      }
      println("output.nDimension() = " + output.nDimension())
      println("output.dim() = " + output.dim())
      println("target.nDimension() = " + target.nDimension())
      println("target.dim() = " + target.dim())

      println("nClasses = " + output.size(output.dim()))*/
      //println("output target size")
      //val size = target.size(1)
      //println("target size1 = " + size + " size2 = " + target.size(2))
      //println("output size1 = " + output.size(1) + " size2 = " + output.size(2))
      /*for (i <- 1 to 10) {
          println( i + " " + target(i))
          println(output(i))
      }*/

      val loss = criterion.forward(output, target).asInstanceOf[Double]
      //println("loss = " + loss)
      val gradOut = criterion.backward(output, target)
      module.backward(input, gradOut)
      (loss, grad)
    }

    val actualModel = specifiedModel()
    val trainSize = 100000
    val testSize = 1000

    val inputs = Tensor[Double](trainSize, 3)
    val r = new scala.util.Random(1)
    inputs.apply1(v => r.nextDouble())



    //println(inputs(1))
    //val targets = actualModel.forward(inputs).resize(Array(trainSize)).apply1(v => 1 + Math.round(v))
    val targets = actualModel.forward(inputs).resize(Array(trainSize, 3)).max(2)._2.squeeze(2)
    println("targets size = " + targets.size(1))

    val trainModel = getTrainModel()
    val criterion = new CrossEntropyCriterion[Double]()
    val (masterWeights, masterGrad) = trainModel.getParameters()
    val optm = new SGD[Double]()
    val config = T("learningRate" -> 10.0, "weightDecay" -> 0.0,
      "momentum" -> 0.0, "learningRateDecay" -> 0.0)
    //println("before training")
    val batchSize = 500
    var epoch = 1
    while (epoch < 100) {
      println("epoch = " + epoch)
      var i = 1
      var l = 0.0
      while (i <= inputs.size(1)) {
        val (grad, loss) = optm.optimize(feval(masterGrad, trainModel, criterion,
          inputs.narrow(1, i, batchSize), targets.narrow(1, i, batchSize)), masterWeights,
          config, config)
        l += loss(0)
        i += batchSize
      }
      println("loss: " + l)
      if (l / inputs.size(1) * batchSize < 6.3e-1) epoch += 1
    }
    //println("after training")
    val testData = Tensor[Double](testSize, 3)
    testData.apply1(v => r.nextDouble())
    //val testTarget = actualModel.forward(testData).apply1(v => Math.round(v))
    val testTarget = actualModel.forward(testData).resize(Array(testSize, 3)).max(2)._2.squeeze(2)

    val testResult = trainModel.forward(testData).max(2)._2.squeeze(2)
    //val testResult = trainModel.forward(testData).apply1(v => 1 + argmax(v))

    var corrects = 0
    var i = 1
    while (i <= testSize) {
      //if (testTarget(Array(i, 1)) == testResult(Array(i, 1))) corrects += 1
      if (testTarget(Array(i)) == testResult(Array(i))) corrects += 1
      i += 1
    }
    println(s"corrects = ${corrects}, testSize = ${testSize}")
    assert(abs(corrects - testSize) / testSize <  0.1 )
  }
}
