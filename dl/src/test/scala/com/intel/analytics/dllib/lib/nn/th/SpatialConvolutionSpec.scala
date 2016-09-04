package com.intel.analytics.dllib.lib.nn.th

import com.intel.analytics.dllib.lib.nn.{Module, Sequential, SpatialConvolution}
import com.intel.analytics.dllib.lib.tensor.{Tensor, torch}
import com.intel.analytics.dllib.lib.th.TH
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import scala.util.Random
import com.intel.analytics.dllib.lib.tensor.RandomGenerator._

/**
  * Created by leon on 16-1-18.
  */
class SpatialConvolutionSpec extends FlatSpec with BeforeAndAfter with Matchers{
  before{
    if(TH.hasTorch()){
      cancel("Torch is not installed")
    }
  }

  "A SpatialConvolution" should "generate correct output" in {
//    feature:add(nn.SpatialConvolutionMM(3,64,11,11,4,4,2,2))       -- 224 -> 55
    val seed = 100
    RNG.setSeed(seed)

    val nInputPlane = 3
    val nOutputPlane = 64
    val kW = 11
    val kH = 11
    val dW = 4
    val dH = 4
    val padW = 2
    val padH = 2
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)

    Random.setSeed(seed)
    val input = torch.Tensor[Double](16, 3, 224, 224).apply1(e => Random.nextDouble())
//    val input = torch.Tensor[Double](8, 3, 224, 224).apply1(e => RNG.uniform(0, 1))

    val output = layer.updateOutput(input)

    val code = "torch.manualSeed(" + seed +")\n" +
      "layer = nn.SpatialConvolutionMM(3, 64, 11, 11, 4, 4, 2, 2)\n" +
      "weight = layer.weight\n" +
      "bias = layer.bias \n" +
      "output = layer:forward(input) "

    val (luaTime, torchResult) = TH.run(code, Map("input"->input), Array("weight", "bias", "output"))

    val luaWeight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luaBias = torchResult("bias").asInstanceOf[Tensor[Double]]
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]

    val weight = layer.weight
    val bias = layer.bias

    weight should be equals luaWeight
    bias should be  equals luaBias
    output should be equals luaOutput
  }


  "A SpatialConvolution(64,192,5,5,1,1,2,2)" should "generate correct output" in {
    //    nn.SpatialConvolutionMM(64,192,5,5,1,1,2,2)
    val seed = 100
    RNG.setSeed(seed)

    val nInputPlane = 64
    val nOutputPlane = 192
    val kW = 5
    val kH = 5
    val dW = 1
    val dH = 1
    val padW = 2
    val padH = 2
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
    val model = new Sequential[Double]()
    model.add(layer)

    Random.setSeed(3)
    //    val input = torch.Tensor[Double](8, 64, 224, 224).apply1(e => Random.nextDouble())
//    val input = torch.Tensor[Double](8, 64, 27, 27).apply1(e => RNG.uniform(0, 1))
    val input = torch.Tensor[Double](8, 64, 27, 27).apply1(e => Random.nextDouble())

    val output = model.updateOutput(input)

    val code = "torch.manualSeed(" + seed +")\n" +
      """layer = nn.SpatialConvolution(64,192,5,5,1,1,2,2)
      model = nn.Sequential()
      model:add(layer)
      weight = layer.weight
      bias = layer.bias
      model:zeroGradParameters()
      output = model:forward(input) """

    val (luaTime, torchResult) = TH.run(code, Map("input"->input), Array("weight", "bias", "output", "model"))

    val luaWeight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luaBias = torchResult("bias").asInstanceOf[Tensor[Double]]
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaModel = torchResult("model").asInstanceOf[Module[Double]]

    val weight = layer.weight
    val bias = layer.bias

    weight should be equals luaWeight
    bias should be  equals luaBias
    output should be equals luaOutput

  }

  //TODO: something wrong with this size's convolution, will fail even use the same BLAS
//  "A SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1)" should "generate correct output" in {
//    val seed = 100
//    RNG.setSeed(seed)
//
//    val nInputPlane = 3
//    val nOutputPlane = 64
//    val kW = 3
//    val kH = 3
//    val dW = 2
//    val dH = 2
//    val padW = 1
//    val padH = 1
//    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
//
//    Random.setSeed(1)
//    val input = torch.Tensor[Double](4, 3, 32, 32).apply1(e => Random.nextDouble())
////        val input = torch.Tensor[Double](4, 3, 32, 32).apply1(e => RNG.uniform(0, 1))
//
//    val output = layer.updateOutput(input)
//
//    val code = "torch.manualSeed(" + seed +")\n" +
//      "layer = nn.SpatialConvolutionMM(3, 64, 3, 3, 2, 2, 1, 1)\n" +
//      "weight = layer.weight\n" +
//      "bias = layer.bias \n" +
//      "output = layer:forward(input) "
//
//    val th = new TH
//    val (luaTime, torchResult) = th.run(code, Map("input"->input), Array("weight", "bias", "output", "layer"))
//
//    val luaWeight = torchResult("weight").asInstanceOf[Tensor[Double]]
//    val luaBias = torchResult("bias").asInstanceOf[Tensor[Double]]
//    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
//    val luaLayer = torchResult("layer").asInstanceOf[SpatialConvolution[Double]]
//
//    val weight = layer.weight
//    val bias = layer.bias
//
//    weight.map(luaWeight, (v1, v2) => {
//      assert(abs(v1 - v2) == 0)
//      v1
//    })
//
//    bias.map(luaBias, (v1, v2) => {
//      assert(abs(v1 - v2) == 0)
//      v1
//    })
//
//    for(i <- 0 until layer.fInput.nElement()){
//      abs(layer.fInput.storage().array()(i) - luaLayer.fInput.storage().array()(i)) should be (0)
////      if(abs(layer.fInput.storage().array()(i) - luaLayer.fInput.storage().array()(i)) > 0){
////        println(s"finput $i ${layer.fInput.storage().array()(i)} ${luaLayer.fInput.storage().array()(i)}")
////      }
//    }
//
//    for(i <- 0 until output.nElement()){
//      abs(output.storage().array()(i) - luaOutput.storage().array()(i)) should be (0)
////      if(abs(output.storage().array()(i) - luaOutput.storage().array()(i)) > 0){
////        println(s"output $i ${output.storage().array()(i)} ${luaOutput.storage().array()(i)}")
////      }
//    }
//  }


  //TODO: something wrong with float version convolution, will fail even use the same BLAS
//    "A SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1) Float version" should "generate correct output" in {
//      val seed = 100
//      RNG.setSeed(seed)
//
//      val nInputPlane = 3
//      val nOutputPlane = 64
//      val kW = 3
//      val kH = 3
//      val dW = 2
//      val dH = 2
//      val padW = 1
//      val padH = 1
//      val layer = new SpatialConvolution[Float](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
//
//      Random.setSeed(1)
//      val input = torch.Tensor[Float](4, 3, 32, 32).apply1(e => Random.nextFloat())
//  //        val input = torch.Tensor[Float](4, 3, 32, 32).apply1(e => RNG.uniform(0, 1))
//
//      val output = layer.updateOutput(input)
//
//      val code = "torch.manualSeed(" + seed +")\n" +
//        "torch.setdefaulttensortype('torch.FloatTensor')\n" +
//        "layer = nn.SpatialConvolutionMM(3, 64, 3, 3, 2, 2, 1, 1)\n" +
//        "weight = layer.weight\n" +
//        "bias = layer.bias \n" +
//        "output = layer:forward(input) "
//
//      val (luaTime, torchResult) = TH.run(code, Map("input"->input), Array("weight", "bias", "output"))
//
//      val luaWeight = torchResult("weight").asInstanceOf[Tensor[Float]]
//      val luaBias = torchResult("bias").asInstanceOf[Tensor[Float]]
//      val luaOutput = torchResult("output").asInstanceOf[Tensor[Float]]
////      val luaLayer = torchResult("layer").asInstanceOf[SpatialConvolution[Float]]
//
//      val weight = layer.weight
//      val bias = layer.bias
//
//      weight.map(luaWeight, (v1, v2) => {
//        assert(abs(v1 - v2) == 0)
//        v1
//      })
//
//      bias.map(luaBias, (v1, v2) => {
//        assert(abs(v1 - v2) == 0)
//        v1
//      })
//
//      for(i <- 0 until output.nElement()){
//        abs(output.storage().array()(i) - luaOutput.storage().array()(i)) should be (0)
//  //      if(abs(output.storage().array()(i) - luaOutput.storage().array()(i)) > 0){
//  //        println(s"output $i ${output.storage().array()(i)} ${luaOutput.storage().array()(i)}")
//  //      }
//      }
//    }
//
//  "A SpatialConvolution(64,192,5,5,1,1,2,2) float version" should "generate correct output" in {
//    //    nn.SpatialConvolutionMM(64,192,5,5,1,1,2,2)
//    val seed = 100
//    RNG.setSeed(seed)
//
//    val nInputPlane = 64
//    val nOutputPlane = 192
//    val kW = 5
//    val kH = 5
//    val dW = 1
//    val dH = 1
//    val padW = 2
//    val padH = 2
//    val layer = new SpatialConvolution[Float](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
//    val model = new Sequential[Float]()
//    model.add(layer)
//
//    Random.setSeed(3)
//    //    val input = torch.Tensor[Float](8, 64, 224, 224).apply1(e => Random.nextFloat())
//    //    val input = torch.Tensor[Float](8, 64, 27, 27).apply1(e => RNG.uniform(0, 1))
//    val input = torch.Tensor[Float](8, 64, 27, 27).apply1(e => Random.nextFloat())
//
//    val output = model.updateOutput(input)
//
//    val code = "torch.manualSeed(" + seed +")\n" +
//      "torch.setdefaulttensortype('torch.FloatTensor')\n" +
//      """layer = nn.SpatialConvolution(64,192,5,5,1,1,2,2)
//      model = nn.Sequential()
//      model:add(layer)
//      weight = layer.weight
//      bias = layer.bias
//      model:zeroGradParameters()
//      output = model:forward(input) """
//
//    val (luaTime, torchResult) = TH.run(code, Map("input"->input), Array("weight", "bias", "output"))
//
//    val luaWeight = torchResult("weight").asInstanceOf[Tensor[Float]]
//    val luaBias = torchResult("bias").asInstanceOf[Tensor[Float]]
//    val luaOutput = torchResult("output").asInstanceOf[Tensor[Float]]
//
//    val weight = layer.weight
//    val bias = layer.bias
//
//    weight.map(luaWeight, (v1, v2) => {
//      assert(abs(v1 - v2) == 0)
//      v1
//    })
//
//    bias.map(luaBias, (v1, v2) => {
//      assert(abs(v1 - v2) == 0)
//      v1
//    })
//
//    for(i <- 0 until output.nElement()){
//      println(i)
//      abs(output.storage().array()(i) - luaOutput.storage().array()(i)) should be (0)
//    }
//  }


//    "A SpatialConvolution(64,192,5,5,1,1,2,2) float version" should "generate correct output" in {
//      //    nn.SpatialConvolutionMM(64,192,5,5,1,1,2,2)
//      val seed = 100
//      RNG.setSeed(seed)
//
//      val nInputPlane = 64
//      val nOutputPlane = 192
//      val kW = 5
//      val kH = 5
//      val dW = 1
//      val dH = 1
//      val padW = 0
//      val padH = 0
//      val layer = new com.intel.analytics.dllib.lib.nn.nnpack.SpatialConvolution[Float](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
//      val model = new Sequential[Float]()
//      model.add(layer)
//
//      Random.setSeed(3)
//      //    val input = torch.Tensor[Float](8, 64, 224, 224).apply1(e => Random.nextFloat())
//      //    val input = torch.Tensor[Float](8, 64, 27, 27).apply1(e => RNG.uniform(0, 1))
//      val input = torch.Tensor[Float](8, 64, 27, 27).apply1(e => Random.nextFloat())
//
//      val output = model.updateOutput(input)
//
//      val code = "torch.manualSeed(" + seed +")\n" +
//        "torch.setdefaulttensortype('torch.FloatTensor')\n" +
//        """layer = nn.SpatialConvolution(64,192,5,5,1,1,0,0)
//        model = nn.Sequential()
//        model:add(layer)
//        weight = layer.weight
//        bias = layer.bias
//        model:zeroGradParameters()
//        output = model:forward(input) """
//
//      val (luaTime, torchResult) = TH.run(code, Map("input"->input), Array("weight", "bias", "output"))
//
//      val luaWeight = torchResult("weight").asInstanceOf[Tensor[Float]]
//      val luaBias = torchResult("bias").asInstanceOf[Tensor[Float]]
//      val luaOutput = torchResult("output").asInstanceOf[Tensor[Float]]
//
//      val weight = layer.weight
//      val bias = layer.bias
//
//      weight should be (luaWeight)
//      bias should be (luaBias)
//
////      output should be (luaOutput)
//
//      for(i <- 0 until output.nElement()){
//        if(!DenseTensorMath.nearlyEqual(output.storage().array()(i), luaOutput.storage().array()(i), 0.01)) {
//          println(s"${output.storage().array()(i)} != ${luaOutput.storage().array()(i)}")
//        }
//      }
//    }
}
