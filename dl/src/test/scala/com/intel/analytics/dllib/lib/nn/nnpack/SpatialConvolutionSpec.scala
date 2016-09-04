package com.intel.analytics.dllib.lib.nn.nnpack

import com.intel.analytics.dllib.lib.th.TH
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class SpatialConvolutionSpec extends FlatSpec with BeforeAndAfter with Matchers{
  before{
    if(TH.hasTorch()){
      cancel("Torch is not installed")
    }
  }

  //TODO: nnpack's data output and gradInput smaller than 1e-4 is a little different with torch.
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
//    val padW = 1
//    val padH = 1
//    val layer = new com.intel.analytics.dllib.lib.nn.nnpack.SpatialConvolution[Float](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
//    val model = new Sequential[Float]()
//    model.add(layer)
//
//    Random.setSeed(3)
//    //    val input = torch.Tensor[Float](8, 64, 224, 224).apply1(e => Random.nextFloat())
//    //    val input = torch.Tensor[Float](8, 64, 27, 27).apply1(e => RNG.uniform(0, 1))
//    val input = torch.Tensor[Float](8, 64, 27, 27).apply1(e => Random.nextFloat())
//    val gradOutput = torch.Tensor[Float](8, 192, 25, 25).apply1(e => Random.nextFloat())
//
//    val output = model.updateOutput(input)
//    val gradInput = model.backward(input, gradOutput)
//
//    val code = "torch.manualSeed(" + seed +")\n" +
//      "torch.setdefaulttensortype('torch.FloatTensor')\n" +
//      s"layer = nn.SpatialConvolution($nInputPlane, $nOutputPlane, $kW, $kH, $dW, $dH, $padW, $padH)" +
//      """ model = nn.Sequential()
//        model:add(layer)
//        weight = layer.weight
//        bias = layer.bias
//        model:zeroGradParameters()
//        output = model:forward(input)
//        gradInput = model:backward(input, gradOutput)
//        gradWeight = layer.gradWeight
//        gradBias = layer.gradBias
//      """
//
//    val (luaTime, torchResult) = TH.run(code, Map("input"->input, "gradOutput"->gradOutput), Array("weight", "bias", "output", "gradInput", "gradWeight", "gradBias"))
//
//    val luaWeight = torchResult("weight").asInstanceOf[Tensor[Float]]
//    val luaBias = torchResult("bias").asInstanceOf[Tensor[Float]]
//    val luaOutput = torchResult("output").asInstanceOf[Tensor[Float]]
//    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Float]]
//    val luaGradBias = torchResult("gradBias").asInstanceOf[Tensor[Float]]
//    val luaGradWeight = torchResult("gradWeight").asInstanceOf[Tensor[Float]]
//
//    val weight = layer.weight
//    val bias = layer.bias
//
//    var i = 0
//    while(i < output.nElement()){
//      if(abs(output.storage().array()(i) - luaOutput.storage().array()(i)) > 1e-6) println( s"${output.storage().array()(i)} != ${luaOutput.storage().array()(i)}")
//      i += 1
//    }
//
////    i = 0
////    while(i < gradInput.nElement()){
//////      if(abs(gradInput.storage().array()(i) - luaGradInput.storage().array()(i)) > 1e-6) println( s"${i} ${gradInput.storage().array()(i)} != ${luaGradInput.storage().array()(i)}")
////      if(!DenseTensorMath.nearlyEqual(gradInput.storage().array()(i), luaGradInput.storage().array()(i), DenseTensorMath.floatEpsilon)) {
////        println( s"${i} ${gradInput.storage().array()(i)} != ${luaGradInput.storage().array()(i)}")
////      }
////      i += 1
////    }
//
//    i = 0
//    while(i < output.nElement()){
//      //      if(abs(gradInput.storage().array()(i) - luaGradInput.storage().array()(i)) > 1e-6) println( s"${i} ${gradInput.storage().array()(i)} != ${luaGradInput.storage().array()(i)}")
//      if(!DenseTensorMath.nearlyEqual(output.storage().array()(i), luaOutput.storage().array()(i), DenseTensorMath.floatEpsilon)) {
//        println( s"${i} ${output.storage().array()(i)} != ${luaOutput.storage().array()(i)}")
//      }
//      i += 1
//    }
//
//    weight should be (luaWeight)
//    bias should be (luaBias)
//    output should be (luaOutput)
//    layer.gradBias should be (luaGradBias)
//    layer.gradWeight should be (luaGradWeight)
//    gradInput should be (luaGradInput)
//  }

  //TODO: nnpack's data output and gradInput smaller than 1e-4 is a little different with spark-nn conv.
//  "A SpatialConvolution(64,192,5,5,1,1,2,2) float version compare with spark conv" should "generate correct output" in {
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
//    val padW = 1
//    val padH = 1
//    val layer = new com.intel.analytics.dllib.lib.nn.nnpack.SpatialConvolution[Float](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
//
//    Random.setSeed(3)
//    //    val input = torch.Tensor[Float](8, 64, 224, 224).apply1(e => Random.nextFloat())
//    //    val input = torch.Tensor[Float](8, 64, 27, 27).apply1(e => RNG.uniform(0, 1))
//    val input = torch.Tensor[Float](8, 64, 27, 27).apply1(e => Random.nextFloat())
//    val gradOutput = torch.Tensor[Float](8, 192, 25, 25).apply1(e => Random.nextFloat())
//
//    val output = layer.forward(input)
//    val gradInput = layer.backward(input, gradOutput)
//
//    val sparkLayer = new com.intel.analytics.dllib.lib.nn.SpatialConvolution[Float](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
//    sparkLayer.weight.copy(layer.weight)
//    sparkLayer.bias.copy(layer.bias)
//
//    val sparkOutput = sparkLayer.forward(input)
//    val sparkGradInput = sparkLayer.backward(input, gradOutput)
//
//    val weight = layer.weight
//    val bias = layer.bias
//
//    var i = 0
//    while(i < output.nElement()){
//      if(abs(output.storage().array()(i) - sparkOutput.storage().array()(i)) > 1e-6) println( s"${output.storage().array()(i)} != ${sparkOutput.storage().array()(i)}")
//      i += 1
//    }
//
////        i = 0
////        while(i < gradInput.nElement()){        i = 0
//        while(i < gradInput.nElement()){
//    //      if(abs(gradInput.storage().array()(i) - luaGradInput.storage().array()(i)) > 1e-6) println( s"${i} ${gradInput.storage().array()(i)} != ${luaGradInput.storage().array()(i)}")
//          if(!DenseTensorMath.nearlyEqual(gradInput.storage().array()(i), sparkGradInput.storage().array()(i), DenseTensorMath.floatEpsilon)) {
//            println( s"${i} ${gradInput.storage().array()(i)} != ${sparkGradInput.storage().array()(i)}")
//          }
//          i += 1
//        }
////    //      if(abs(gradInput.storage().array()(i) - luaGradInput.storage().array()(i)) > 1e-6) println( s"${i} ${gradInput.storage().array()(i)} != ${luaGradInput.storage().array()(i)}")
////          if(!DenseTensorMath.nearlyEqual(gradInput.storage().array()(i), sparkGradInput.storage().array()(i), DenseTensorMath.floatEpsilon)) {
////            println( s"${i} ${gradInput.storage().array()(i)} != ${sparkGradInput.storage().array()(i)}")
////          }
////          i += 1
////        }
//
//    i = 0
//    while(i < output.nElement()){
//      //      if(abs(gradInput.storage().array()(i) - luaGradInput.storage().array()(i)) > 1e-6) println( s"${i} ${gradInput.storage().array()(i)} != ${luaGradInput.storage().array()(i)}")
//      if(!DenseTensorMath.nearlyEqual(output.storage().array()(i), sparkOutput.storage().array()(i), DenseTensorMath.floatEpsilon)) {
//        println( s"${i} ${output.storage().array()(i)} != ${sparkOutput.storage().array()(i)}")
//      }
//      i += 1
//    }
//
//    weight should be (sparkLayer.weight)
//    bias should be (sparkLayer.bias)
//    output should be (sparkLayer.output)
//    layer.gradBias should be (sparkLayer.gradBias)
//    layer.gradWeight should be (sparkLayer.gradWeight)
//    gradInput should be (sparkLayer.gradInput)
//  }
}
