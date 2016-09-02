package com.intel.webscaleml.nn.nn.th

import java.util.HashMap

import com.intel.webscaleml.nn.example.{Cifar, GoogleNet}
import com.intel.webscaleml.nn.nn.{ClassNLLCriterion, Module}
import com.intel.webscaleml.nn.optim.SGD
import com.intel.webscaleml.nn.tensor._
import com.intel.webscaleml.nn.th.TH
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.math._
import scala.sys.process._
import scala.util.Random

class CifarVggSpec extends FlatSpec with BeforeAndAfter with Matchers{
  before{
    if(TH.hasTorch()){
      cancel("Torch is not installed")
    }
  }
    //TODO: qiuxin, complete this later.

//    "cifar+bn" should "generate correct output" in {
//      Random.setSeed(3)
//      val input = torch.Tensor[Float](4, 3, 32, 32).apply1(e => Random.nextFloat())
//      val labels = torch.Tensor[Float](4).apply1(e => Random.nextInt(10))
//
//      val seed = 100
//      RNG.setSeed(seed)
//      val model = Cifar.getModel[Float](10, "vggBn")
//
//      model.zeroGradParameters()
//      val parameters = model.getParameters()._1.asInstanceOf[Tensor[Float]]
//      val gradparameters = model.getParameters()._2.asInstanceOf[Tensor[Float]]
//      println(s"model size: ${parameters.nElement()}")
//
//      val code =  "torch.manualSeed(" + seed +")\n" +
//        """
//          torch.setdefaulttensortype('torch.FloatTensor')
//          local nClasses = 10
//          local vgg = nn.Sequential()
//
//          -- building block
//          local function ConvBNReLU(nInputPlane, nOutputPlane)
//            vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
//            vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
//            vgg:add(nn.ReLU(true))
//            return vgg
//          end
//
//          -- Will use "ceil" MaxPooling because we want to save as much
//          -- space as we can
//          local MaxPooling = nn.SpatialMaxPooling
//
//          ConvBNReLU(3,64)
//          ConvBNReLU(64,64)
//          vgg:add(MaxPooling(2,2,2,2):ceil())
//
//          ConvBNReLU(64,128)
//          ConvBNReLU(128,128)
//          vgg:add(MaxPooling(2,2,2,2):ceil())
//
//          ConvBNReLU(128,256)
//          ConvBNReLU(256,256)
//          ConvBNReLU(256,256)
//          vgg:add(MaxPooling(2,2,2,2):ceil())
//
//          ConvBNReLU(256,512)
//          ConvBNReLU(512,512)
//          ConvBNReLU(512,512)
//          vgg:add(MaxPooling(2,2,2,2):ceil())
//
//          ConvBNReLU(512,512)
//          ConvBNReLU(512,512)
//          ConvBNReLU(512,512)
//          vgg:add(MaxPooling(2,2,2,2):ceil())
//          vgg:add(nn.View(512))
//
//          classifier = nn.Sequential()
//          classifier:add(nn.Linear(512,512))
//          classifier:add(nn.BatchNormalization(512))
//          classifier:add(nn.ReLU(true))
//          classifier:add(nn.Linear(512,nClasses))
//          classifier:add(nn.LogSoftMax())
//          vgg:add(classifier)
//          local model = vgg
//
//          parameters, gradParameters = model:getParameters()
//          model:zeroGradParameters()
//          parameters_initial = parameters : clone()
//          gradParameters_initial = gradParameters : clone()
//
//          criterion =  nn.ClassNLLCriterion()
//
//          state = {
//            learningRate = 1e-2,
//            momentum = 0.9,
//            dampening = 0.0,
//            weightDecay = 5e-4
//          }
//
//          feval = function(x)
//            model:zeroGradParameters()
//            model_initial = model : clone()
//
//            local output1 = model:forward(input)
//            local err1 = criterion:forward(output1, labels)
//            local gradOutput1 = criterion:backward(output1, labels)
//            model:backward(input, gradOutput1)
//            return err1, gradParameters
//          end
//
//          for i = 1,1,1 do
//            w, err = optim.sgd(feval, parameters, state)
//          end
//
//          output=model.output
//          gradOutput=criterion.gradInput
//          gradInput = model.gradInput
//
//          weights, grad = model:getParameters()
//        """
//
//      val (luaTime, torchResult) = TH.run(code, Map("input"->input, "labels"->labels), Array("output", "gradOutput","err","parameters_initial","gradParameters_initial", "grad", "weights"))
//      val outputTorch = torchResult("output").asInstanceOf[Tensor[Float]]
//      val errTorch = torchResult("err").asInstanceOf[HashMap[Double, Double]].get(1.0)
//      val parametersInitTorch = torchResult("parameters_initial").asInstanceOf[Tensor[Float]]
//      val gradGarametersInitTorch = torchResult("gradParameters_initial").asInstanceOf[Tensor[Float]]
//      val gradTorch = torchResult("grad").asInstanceOf[Tensor[Float]]
//      val weightsTorch = torchResult("weights").asInstanceOf[Tensor[Float]]
//      val gradOutputTorch = torchResult("gradOutput").asInstanceOf[Tensor[Float]]
////      val modelTorch = torchResult("model").asInstanceOf[Module[Float]]
//
//      require(parameters == parametersInitTorch , "parameter compare failed")
//
//      require(gradparameters == gradGarametersInitTorch, "gradparameter compare failed")
//
//      val (weights, grad) = model.getParameters()
//      val criterion = new ClassNLLCriterion[Float]()
//
//      val state = T("learningRate" -> 1e-2, "momentum" -> 0.9, "weightDecay" -> 5e-4, "dampening" -> 0.0)
//
//      val sgd = new SGD[Float]
//
////      for(i <- 1 to 99) {
////        model.zeroGradParameters()
////        val outputtest = model.forward(input)
////        val loss = criterion.forward(outputtest, labels)
////        val gradoutputtest = criterion.backward(outputtest, labels)
////        model.backward(input, gradoutputtest)
////        sgd.optimize(_ => (loss, grad), weights, state, state)
////      }
//
//      model.zeroGradParameters()
//      val outputTest = model.forward(input)
//      val errTest = criterion.forward(outputTest, labels)
//      val gradOutputTest = criterion.backward(outputTest, labels)
//      model.backward(input, gradOutputTest)
//      sgd.optimize(_ => (errTest, grad), weights, state, state)
//
//
////      for(i <- 0 until model.modules.size){
////        println(i)
////        if(modelTorch.modules(i).gradWeight != null) {
////          val offset = model.modules(i).gradWeight.storageOffset() - 1
////          val offsetTorch = modelTorch.modules(i).gradWeight.storageOffset() - 1
////          for(j <- 0 until model.modules(i).gradWeight.nElement()){
////            if (abs(model.modules(i).gradWeight.storage().array()(j + offset) - modelTorch.modules(i).gradWeight.storage().array()(j + offsetTorch)) > 1e-13) {
////              println(s"$i $j ${model.modules(i).gradWeight.storage().array()(j + offset)} ${modelTorch.modules(i).gradWeight.storage().array()(j + offsetTorch)}")
////            }
////          }
////        }
////      }
//
//      var j = 0
//      var gradAbs = 0.0
//      while(j < grad.nElement()){
//        val tmp = abs(grad.storage().array()(j) - gradTorch.storage().array()(j))
//        gradAbs += tmp
//        if (tmp > 1e-4) println(s"$j: ${grad.storage().array()(j)} ${gradTorch.storage().array()(j)}")
//        j += 1
//      }
//      //    assert(gradAbs < 1e-9)
//      println(s"gradAbs:$gradAbs")
//
//      gradOutputTest should be (gradOutputTorch)
//
//      var parametersAbs = 0.0
//      j = 0
//      while(j < weights.nElement()){
//        val tmp = abs(weights.storage().array()(j) - weightsTorch.storage().array()(j))
//        parametersAbs += tmp
//        //      if (tmp > 1e-10) println(s"$j: ${gradInput.storage().array()(j)} ${gradInputTorch.storage().array()(j)}")
//        j += 1
//      }
//      //    assert(parametersAbs < 2e-15)
//      println(s"parametersAbs${weights.nElement()}:$parametersAbs")
//
//      var outputAbs = 0.0
//      outputTest.map(outputTorch, (v1, v2) => {
//        outputAbs += abs(v1 - v2)
//        //      if(v1 != v2) println(s"$v1 $v2")
//        //      assert(abs(v1 - v2) < 1e-14);
//        v1
//      })
//      //    assert(outputAbs < 1e-11)
//      println(s"outputAbs:$outputAbs")
//
//      println(s"err:${abs(errTest - errTorch)}")
//      assert(abs(errTest - errTorch) < 2e-6)
//
//    }

//  "cifar+bn" should "generate correct output" in {
//    Random.setSeed(3)
//    val input = torch.Tensor[Double](4, 3, 32, 32).apply1(e => Random.nextDouble())
//    val labels = torch.Tensor[Double](4).apply1(e => Random.nextInt(10))
//
//    val seed = 100
//    RNG.setSeed(seed)
//    val model = Cifar.getModel[Double](10, "vggBn")
//
//    model.zeroGradParameters()
//    val parameters = model.getParameters()._1.asInstanceOf[Tensor[Double]]
//    val gradparameters = model.getParameters()._2.asInstanceOf[Tensor[Double]]
//    println(s"model size: ${parameters.nElement()}")
//
//    val code =  "torch.manualSeed(" + seed +")\n" +
//      """
//        local nClasses = 10
//        local vgg = nn.Sequential()
//
//        -- building block
//        local function ConvBNReLU(nInputPlane, nOutputPlane)
//          vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
//          vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
//          vgg:add(nn.ReLU(true))
//          return vgg
//        end
//
//        -- Will use "ceil" MaxPooling because we want to save as much
//        -- space as we can
//        local MaxPooling = nn.SpatialMaxPooling
//
//        ConvBNReLU(3,64)
//        ConvBNReLU(64,64)
//        vgg:add(MaxPooling(2,2,2,2):ceil())
//
//        ConvBNReLU(64,128)
//        ConvBNReLU(128,128)
//        vgg:add(MaxPooling(2,2,2,2):ceil())
//
//        ConvBNReLU(128,256)
//        ConvBNReLU(256,256)
//        ConvBNReLU(256,256)
//        vgg:add(MaxPooling(2,2,2,2):ceil())
//
//        ConvBNReLU(256,512)
//        ConvBNReLU(512,512)
//        ConvBNReLU(512,512)
//        vgg:add(MaxPooling(2,2,2,2):ceil())
//
//        ConvBNReLU(512,512)
//        ConvBNReLU(512,512)
//        ConvBNReLU(512,512)
//        vgg:add(MaxPooling(2,2,2,2):ceil())
//        vgg:add(nn.View(512))
//
//        classifier = nn.Sequential()
//        classifier:add(nn.Linear(512,512))
//        classifier:add(nn.BatchNormalization(512))
//        classifier:add(nn.ReLU(true))
//        classifier:add(nn.Linear(512,nClasses))
//        classifier:add(nn.LogSoftMax())
//        vgg:add(classifier)
//        local model = vgg
//
//        parameters, gradParameters = model:getParameters()
//        model:zeroGradParameters()
//        parameters_initial = parameters : clone()
//        gradParameters_initial = gradParameters : clone()
//
//        criterion =  nn.ClassNLLCriterion()
//
//        state = {
//          learningRate = 1e-2,
//          momentum = 0.9,
//          dampening = 0.0,
//          weightDecay = 5e-4
//        }
//
//        feval = function(x)
//          model:zeroGradParameters()
//          model_initial = model : clone()
//
//          local output1 = model:forward(input)
//          local err1 = criterion:forward(output1, labels)
//          local gradOutput1 = criterion:backward(output1, labels)
//          model:backward(input, gradOutput1)
//          return err1, gradParameters
//        end
//
//        for i = 1,100,1 do
//          model:zeroGradParameters()
//          w, err = optim.sgd(feval, parameters, state)
//        end
//
//        output=model.output
//        gradOutput=criterion.gradInput
//        gradInput = model.gradInput
//
//        parameters, gradParameters = model:getParameters()
//      """
//
//    val th = new TH
//    val (luaTime, torchResult) = th.run(code, Map("input"->input, "labels"->labels), Array("output", "gradOutput","err","parameters_initial","gradParameters_initial", "gradParameters", "parameters", "model"))
//    val outputTorch = torchResult("output").asInstanceOf[Tensor[Double]]
//    val errTorch = torchResult("err").asInstanceOf[HashMap[Double, Double]].get(1.0)
//    val parametersInitTorch = torchResult("parameters_initial").asInstanceOf[Tensor[Double]]
//    val gradGarametersInitTorch = torchResult("gradParameters_initial").asInstanceOf[Tensor[Double]]
//    val gradParametersTorch = torchResult("gradParameters").asInstanceOf[Tensor[Double]]
//    val gradOutputTorch = torchResult("gradOutput").asInstanceOf[Tensor[Double]]
//    val modelTorch = torchResult("model").asInstanceOf[Module[Double]]
//
//    require(parameters == parametersInitTorch , "parameter compare failed")
//
//    require(gradparameters == gradGarametersInitTorch, "gradparameter compare failed")
//
//    val (weights, grad) = model.getParameters()
//    val criterion = new ClassNLLCriterion[Double]()
//
//    val state = T("learningRate" -> 1e-2, "momentum" -> 0.9, "weightDecay" -> 5e-4, "dampening" -> 0.0)
//
//    val sgd = new SGD[Double]
//
//    val epsilon = System.getProperty("DoubleTensorEpsilon", "0.0000001").toDouble
//
//    for(i <- 1 to 99) {
//      model.zeroGradParameters()
//      val outputtest = model.forward(input)
//      val loss = criterion.forward(outputtest, labels)
//      val gradoutputtest = criterion.backward(outputtest, labels)
//      model.backward(input, gradoutputtest)
//      sgd.optimize(_ => (loss, grad), weights, state, state)
//    }
//
//    model.zeroGradParameters()
//    val outputTest = model.forward(input)
//    val errTest = criterion.forward(outputTest, labels)
//    val gradOutputTest = criterion.backward(outputTest, labels)
//    model.backward(input, gradOutputTest)
//    sgd.optimize(_ => (errTest, grad), weights, state, state)
//
//
//    for(i <- 0 until model.modules.size){
//      println(i)
//      if(modelTorch.modules(i).gradWeight != null) {
//        val offset = model.modules(i).gradWeight.storageOffset() - 1
//        val offsetTorch = modelTorch.modules(i).gradWeight.storageOffset() - 1
//        for(j <- 0 until model.modules(i).gradWeight.nElement()){
//          if (abs(model.modules(i).gradWeight.storage().array()(j + offset) - modelTorch.modules(i).gradWeight.storage().array()(j + offsetTorch)) > 1e-13) {
//            println(s"$i $j ${model.modules(i).gradWeight.storage().array()(j + offset)} ${modelTorch.modules(i).gradWeight.storage().array()(j + offsetTorch)}")
//          }
//        }
//      }
//    }
//
//
//
//
//    var j = 0
//    var gradAbs = 0.0
//    while(j < grad.nElement()){
//      val tmp = abs(grad.storage().array()(j) - gradParametersTorch.storage().array()(j))
//      gradAbs += tmp
////      if (tmp > 1e-14) println(s"$j: ${grad.storage().array()(j)} ${gradParametersTorch.storage().array()(j)}")
//      j += 1
//    }
//    //    assert(gradAbs < 1e-9)
//    println(s"gradAbs:$gradAbs")
//
//    val parametersTorch = torchResult("parameters").asInstanceOf[Tensor[Double]]
//
//    gradOutputTest should be (gradOutputTorch)
//
//    var parametersAbs = 0.0
//    j = 0
//    while(j < weights.nElement()){
//      val tmp = abs(weights.storage().array()(j) - parametersTorch.storage().array()(j))
//      parametersAbs += tmp
//      //      if (tmp > 1e-10) println(s"$j: ${gradInput.storage().array()(j)} ${gradInputTorch.storage().array()(j)}")
//      j += 1
//    }
//    //    assert(parametersAbs < 2e-15)
//    println(s"parametersAbs${weights.nElement()}:$parametersAbs")
//
//    var outputAbs = 0.0
//    outputTest.map(outputTorch, (v1, v2) => {
//      outputAbs += abs(v1 - v2)
//      //      if(v1 != v2) println(s"$v1 $v2")
//      //      assert(abs(v1 - v2) < 1e-14);
//      v1
//    })
//    //    assert(outputAbs < 1e-11)
//    println(s"outputAbs:$outputAbs")
//
//    println(s"err:${abs(errTest - errTorch)}")
//    assert(abs(errTest - errTorch) == 0)
//
//  }
//
//
//
//  "VggBn evaluate" should "generate correct output" in {
//    Random.setSeed(3)
//    val input = torch.Tensor[Double](4, 3, 32, 32).apply1(e => Random.nextDouble())
//    val labels = torch.Tensor[Double](4).apply1(e => Random.nextInt(10))
//
//    val seed = 100
//    RNG.setSeed(seed)
//    val model = Cifar.getModel[Double](10, "vggBn")
//
//    model.zeroGradParameters()
//    val parameters = model.getParameters()._1.asInstanceOf[Tensor[Double]]
//    val gradparameters = model.getParameters()._2.asInstanceOf[Tensor[Double]]
//    println(s"model size: ${parameters.nElement()}")
//
//    val code =  "torch.manualSeed(" + seed +")\n" +
//      """
//        local nClasses = 10
//        local vgg = nn.Sequential()
//
//        -- building block
//        local function ConvBNReLU(nInputPlane, nOutputPlane)
//          vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
//          vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
//          vgg:add(nn.ReLU(true))
//          return vgg
//        end
//
//        -- Will use "ceil" MaxPooling because we want to save as much
//        -- space as we can
//        local MaxPooling = nn.SpatialMaxPooling
//
//        ConvBNReLU(3,64)
//        ConvBNReLU(64,64)
//        vgg:add(MaxPooling(2,2,2,2):ceil())
//
//        ConvBNReLU(64,128)
//        ConvBNReLU(128,128)
//        vgg:add(MaxPooling(2,2,2,2):ceil())
//
//        ConvBNReLU(128,256)
//        ConvBNReLU(256,256)
//        ConvBNReLU(256,256)
//        vgg:add(MaxPooling(2,2,2,2):ceil())
//
//        ConvBNReLU(256,512)
//        ConvBNReLU(512,512)
//        ConvBNReLU(512,512)
//        vgg:add(MaxPooling(2,2,2,2):ceil())
//
//        ConvBNReLU(512,512)
//        ConvBNReLU(512,512)
//        ConvBNReLU(512,512)
//        vgg:add(MaxPooling(2,2,2,2):ceil())
//        vgg:add(nn.View(512))
//
//        classifier = nn.Sequential()
//        classifier:add(nn.Linear(512,512))
//        classifier:add(nn.BatchNormalization(512))
//        classifier:add(nn.ReLU(true))
//        classifier:add(nn.Linear(512,nClasses))
//        classifier:add(nn.LogSoftMax())
//        vgg:add(classifier)
//        local model = vgg
//
//        local parameters, gradParameters = model:getParameters()
//        model:zeroGradParameters()
//        parameters_initial = parameters : clone()
//        gradParameters_initial = gradParameters : clone()
//
//        criterion =  nn.ClassNLLCriterion()
//
//        state = {
//          learningRate = 1e-2,
//          momentum = 0.9,
//          dampening = 0.0,
//          weightDecay = 5e-4
//        }
//
//        feval = function(x)
//          model:zeroGradParameters()
//          model_initial = model : clone()
//
//          local output1 = model:forward(input)
//          local err1 = criterion:forward(output1, labels)
//          local gradOutput1 = criterion:backward(output1, labels)
//          model:backward(input, gradOutput1)
//          return err1, gradParameters
//        end
//
//        for i = 1,1,1 do
//          w, err = optim.sgd(feval, parameters, state)
//        end
//
//        output=model.output
//        gradOutput=criterion.gradInput
//        gradInput = model.gradInput
//        model:evaluate()
//        output = model:forward(input)
//      """
//
//    val th = new TH
//    val (luaTime, torchResult) = th.run(code, Map("input"->input, "labels"->labels), Array("output", "gradOutput","err","parameters_initial","gradParameters_initial", "gradInput", "parameters", "model"))
//    val outputTorch = torchResult("output").asInstanceOf[Tensor[Double]]
//    val errTorch = torchResult("err").asInstanceOf[HashMap[Double, Double]].get(1.0)
//    val parameterTorch = torchResult("parameters_initial").asInstanceOf[Tensor[Double]]
//    val gradparameterTorch = torchResult("gradParameters_initial").asInstanceOf[Tensor[Double]]
//    val modelTorch = torchResult("model").asInstanceOf[Module[Double]]
//    val gradInputTorch = torchResult("gradInput").asInstanceOf[Tensor[Double]]
//    val gradOutputTorch = torchResult("gradOutput").asInstanceOf[Tensor[Double]]
//
//    require(parameters == parameterTorch , "parameter compare failed")
//
//    require(gradparameters == gradparameterTorch, "gradparameter compare failed")
//
//    val (weights, grad) = model.getParameters()
//    val criterion = new ClassNLLCriterion[Double]()
//
//    val state = T("learningRate" -> 1e-2, "momentum" -> 0.9, "weightDecay" -> 5e-4, "dampening" -> 0.0)
//
//    val sgd = new SGD[Double]
//
//    val epsilon = System.getProperty("DoubleTensorEpsilon", "0.0000001").toDouble
//
//    //    for(i <- 1 to 4) {
//    //      model.zeroGradParameters()
//    //      val outputtest = model.forward(input)
//    //      val loss = criterion.forward(outputtest, labels)
//    //      val gradoutputtest = criterion.backward(outputtest, labels)
//    //      model.backward(input, gradoutputtest)
//    //      sgd.optimize(_ => (loss, grad), weights, state, state)
//    //    }
//
//    model.zeroGradParameters()
//    val outputTest = model.forward(input)
//    val errTest = criterion.forward(outputTest, labels)
//    val gradOutputTest = criterion.backward(outputTest, labels)
//    val gradInput = model.backward(input, gradOutputTest)
//    sgd.optimize(_ => (errTest, grad), weights, state, state)
//
////    for(i <- 0 until model.modules.size){
////      println(i)
////      for(j <- 0 until model.modules(i).output.nElement()){
////        if(model.modules(i).output.storage().array()(j) != modelTorch.modules(i).output.storage().array()(j)){
////          println(s"$i $j ${model.modules(i).output.storage().array()(j)} ${modelTorch.modules(i).output.storage().array()(j)}")
////        }
////      }
////    }
////
////    for(i <- 0 until model.modules(45).modules.size){
////      println(i)
////      for(j <- 0 until model.modules(45).modules(i).output.nElement()){
////        if(model.modules(45).modules(i).output.storage().array()(j) != modelTorch.modules(45).modules(i).output.storage().array()(j)){
////          println(s"$i $j ${model.modules(45).modules(i).output.storage().array()(j)} ${modelTorch.modules(45).modules(i).output.storage().array()(j)}")
////        }
////      }
////    }
//
//    var j = 0
//    var gradInputAbs = 0.0
//    while(j < gradInput.nElement()){
//      val tmp = abs(gradInput.storage().array()(j) - gradInputTorch.storage().array()(j))
//      gradInputAbs += tmp
//      //            if (tmp > 0) println(s"$j: ${gradInput.storage().array()(j)} ${gradInputTorch.storage().array()(j)}")
//      j += 1
//    }
//    //    assert(gradInputAbs < 2e-20)
//    println(s"gradInputAbs:$gradInputAbs")
//
//    val parametersTorch = torchResult("parameters").asInstanceOf[Tensor[Double]]
//
//    gradOutputTest should be (gradOutputTorch)
//
//    var parametersAbs = 0.0
//    while(j < parameters.nElement()){
//      val tmp = abs(parameters.storage().array()(j) - parametersTorch.storage().array()(j))
//      parametersAbs += tmp
//      //      if (tmp > 1e-10) println(s"$j: ${gradInput.storage().array()(j)} ${gradInputTorch.storage().array()(j)}")
//      j += 1
//    }
//    //    assert(parametersAbs < 2e-15)
//    println(s"parametersAbs:$parametersAbs")
//
//    var outputAbs = 0.0
//    outputTest.map(outputTorch, (v1, v2) => {
//      outputAbs += abs(v1 - v2)
//      //            if(v1 != v2) println(s"$v1 $v2")
//      //      assert(abs(v1 - v2) < 1e-14);
//      v1
//    })
//    //    assert(outputAbs < 2e-14)
//    println(s"outputAbs:$outputAbs")
//
//    println(s"err:${abs(errTest - errTorch)}")
//    assert(abs(errTest - errTorch) == 0)
//
//  }

//  "VggBn" should "generate correct output" in {
//    Random.setSeed(3)
//    val input = torch.Tensor[Double](4, 3, 32, 32).apply1(e => Random.nextDouble())
//    val labels = torch.Tensor[Double](4).apply1(e => Random.nextInt(10))
//
//    val seed = 100
//    RNG.setSeed(seed)
//    val model = Cifar.getModel(10, "vggBn")
//
//    model.zeroGradParameters()
//    val parameters = model.getParameters()._1.asInstanceOf[Tensor[Double]]
//    val gradparameters = model.getParameters()._2.asInstanceOf[Tensor[Double]]
//    println(s"model size: ${parameters.nElement()}")
//
//    val code =  "torch.manualSeed(" + seed +")\n" +
//      """
//        local nClasses = 10
//        local vgg = nn.Sequential()
//
//        -- building block
//        local function ConvBNReLU(nInputPlane, nOutputPlane)
//          vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
//          vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
//          vgg:add(nn.ReLU(true))
//          return vgg
//        end
//
//        -- Will use "ceil" MaxPooling because we want to save as much
//        -- space as we can
//        local MaxPooling = nn.SpatialMaxPooling
//
//        ConvBNReLU(3,64)
//        ConvBNReLU(64,64)
//        vgg:add(MaxPooling(2,2,2,2):ceil())
//
//        ConvBNReLU(64,128)
//        ConvBNReLU(128,128)
//        vgg:add(MaxPooling(2,2,2,2):ceil())
//
//        ConvBNReLU(128,256)
//        ConvBNReLU(256,256)
//        ConvBNReLU(256,256)
//        vgg:add(MaxPooling(2,2,2,2):ceil())
//
//        ConvBNReLU(256,512)
//        ConvBNReLU(512,512)
//        ConvBNReLU(512,512)
//        vgg:add(MaxPooling(2,2,2,2):ceil())
//
//        ConvBNReLU(512,512)
//        ConvBNReLU(512,512)
//        ConvBNReLU(512,512)
//        vgg:add(MaxPooling(2,2,2,2):ceil())
//        vgg:add(nn.View(512))
//
//        classifier = nn.Sequential()
//        classifier:add(nn.Linear(512,512))
//        classifier:add(nn.BatchNormalization(512))
//        classifier:add(nn.ReLU(true))
//        classifier:add(nn.Linear(512,nClasses))
//        classifier:add(nn.LogSoftMax())
//        vgg:add(classifier)
//        local model = vgg
//
//        local parameters, gradParameters = model:getParameters()
//        model:zeroGradParameters()
//        parameters_initial = parameters : clone()
//        gradParameters_initial = gradParameters : clone()
//
//        criterion =  nn.ClassNLLCriterion()
//
//        state = {
//          learningRate = 1e-2,
//          momentum = 0.9,
//          dampening = 0.0,
//          weightDecay = 5e-4
//        }
//
//        feval = function(x)
//          model:zeroGradParameters()
//          model_initial = model : clone()
//
//          local output1 = model:forward(input)
//          local err1 = criterion:forward(output1, labels)
//          local gradOutput1 = criterion:backward(output1, labels)
//          model:backward(input, gradOutput1)
//          return err1, gradParameters
//        end
//
//        for i = 1,1,1 do
//          w, err = optim.sgd(feval, parameters, state)
//        end
//
//        output=model.output
//        gradOutput=criterion.gradInput
//        gradInput = model.gradInput
//
//      """
//
//    val th = new TH
//    val (luaTime, torchResult) = th.run(code, Map("input"->input, "labels"->labels), Array("output", "gradOutput","err","parameters_initial","gradParameters_initial", "gradInput", "parameters", "model"))
//    val outputTorch = torchResult("output").asInstanceOf[Tensor[Double]]
//    val errTorch = torchResult("err").asInstanceOf[HashMap[Double, Double]].get(1.0)
//    val parameterTorch = torchResult("parameters_initial").asInstanceOf[Tensor[Double]]
//    val gradparameterTorch = torchResult("gradParameters_initial").asInstanceOf[Tensor[Double]]
//    val modelTorch = torchResult("model").asInstanceOf[Module[Double]]
//    val gradInputTorch = torchResult("gradInput").asInstanceOf[Tensor[Double]]
//    val gradOutputTorch = torchResult("gradOutput").asInstanceOf[Tensor[Double]]
//
//    require(parameters == parameterTorch , "parameter compare failed")
//
//    require(gradparameters == gradparameterTorch, "gradparameter compare failed")
//
//    val (weights, grad) = model.getParameters()
//    val criterion = new ClassNLLCriterion[Double]()
//
//    val state = T("learningRate" -> 1e-2, "momentum" -> 0.9, "weightDecay" -> 5e-4, "dampening" -> 0.0)
//
//    val sgd = new SGD[Double]
//
//    val epsilon = System.getProperty("DoubleTensorEpsilon", "0.0000001").toDouble
//
////    for(i <- 1 to 4) {
////      model.zeroGradParameters()
////      val outputtest = model.forward(input)
////      val loss = criterion.forward(outputtest, labels)
////      val gradoutputtest = criterion.backward(outputtest, labels)
////      model.backward(input, gradoutputtest)
////      sgd.optimize(_ => (loss, grad), weights, state, state)
////    }
//
//    model.zeroGradParameters()
//    val outputTest = model.forward(input)
//    val errTest = criterion.forward(outputTest, labels)
//    val gradOutputTest = criterion.backward(outputTest, labels)
//    val gradInput = model.backward(input, gradOutputTest)
//    sgd.optimize(_ => (errTest, grad), weights, state, state)
//
////    for(i <- 0 until model.modules.size){
////      for(j <- 0 until model.modules(i).gradInput.nElement()){
////        if(model.modules(i).gradInput.storage().array()(j) != modelTorch.modules(i).gradInput.storage().array()(j)){
////          println(s"$i $j ${model.modules(i).gradInput.storage().array()(j)} ${modelTorch.modules(i).gradInput.storage().array()(j)}")
////        }
////      }
////    }
////    for(i <- 0 until model.modules.size){
////      for(j <- 0 until model.modules(i).output.nElement()){
////        if(model.modules(i).output.storage().array()(j) != modelTorch.modules(i).output.storage().array()(j)){
////          println(s"$i $j ${model.modules(i).output.storage().array()(j)} ${modelTorch.modules(i).output.storage().array()(j)}")
////        }
////      }
////    }
////
////    for(i <- 0 until model.modules(45).modules.size){
////      println(i)
////      for(j <- 0 until model.modules(45).modules(i).output.nElement()){
////        if(model.modules(45).modules(i).output.storage().array()(j) != modelTorch.modules(45).modules(i).output.storage().array()(j)){
////          println(s"$i $j ${model.modules(45).modules(i).output.storage().array()(j)} ${modelTorch.modules(45).modules(i).output.storage().array()(j)}")
////        }
////      }
////    }
//
//    var j = 0
//    var gradInputAbs = 0.0
//    while(j < gradInput.nElement()){
//      val tmp = abs(gradInput.storage().array()(j) - gradInputTorch.storage().array()(j))
//      gradInputAbs += tmp
////            if (tmp > 0) println(s"$j: ${gradInput.storage().array()(j)} ${gradInputTorch.storage().array()(j)}")
//      j += 1
//    }
//    assert(gradInputAbs < 2e-11)
//    println(s"gradInputAbs:$gradInputAbs")
//
//    val parametersTorch = torchResult("parameters").asInstanceOf[Tensor[Double]]
//
//    gradOutputTest should be (gradOutputTorch)
//
//    var parametersAbs = 0.0
//    while(j < parameters.nElement()){
//      val tmp = abs(parameters.storage().array()(j) - parametersTorch.storage().array()(j))
//      parametersAbs += tmp
//      //      if (tmp > 1e-10) println(s"$j: ${gradInput.storage().array()(j)} ${gradInputTorch.storage().array()(j)}")
//      j += 1
//    }
//    assert(parametersAbs < 2)
//    println(s"parametersAbs:$parametersAbs")
//
//    var outputAbs = 0.0
//    outputTest.map(outputTorch, (v1, v2) => {
//      outputAbs += abs(v1 - v2)
////            if(v1 != v2) println(s"$v1 $v2")
//      //      assert(abs(v1 - v2) < 1e-14);
//      v1
//    })
//    assert(outputAbs < 1e-13)
//    println(s"outputAbs:$outputAbs")
//
//    println(s"err:${abs(errTest - errTorch)}")
//    assert(abs(errTest - errTorch) < 2e-15)
//
//  }
//
//  "VggBnDrop" should "generate correct output" in {
//    Random.setSeed(3)
//    val input = torch.Tensor[Double](4, 3, 32, 32).apply1(e => Random.nextDouble())
//    val labels = torch.Tensor[Double](4).apply1(e => Random.nextInt(10))
//
//    val seed = 100
//    RNG.setSeed(seed)
//    val model = Cifar.getModel(10, "vggBnDo")
//
//    model.zeroGradParameters()
//    val parameters = model.getParameters()._1.asInstanceOf[Tensor[Double]]
//    val gradparameters = model.getParameters()._2.asInstanceOf[Tensor[Double]]
//    println(s"model size: ${parameters.nElement()}")
//
//    val code =  "torch.manualSeed(" + seed +")\n" +
//      """
//        local nClasses = 10
//        local vgg = nn.Sequential()
//
//        -- building block
//        local function ConvBNReLU(nInputPlane, nOutputPlane)
//          vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
//          vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
//          vgg:add(nn.ReLU(true))
//          return vgg
//        end
//
//        -- Will use "ceil" MaxPooling because we want to save as much
//        -- space as we can
//        local MaxPooling = nn.SpatialMaxPooling
//
//        ConvBNReLU(3,64):add(nn.Dropout(0.3))
//        ConvBNReLU(64,64)
//        vgg:add(MaxPooling(2,2,2,2):ceil())
//
//        ConvBNReLU(64,128):add(nn.Dropout(0.4))
//        ConvBNReLU(128,128)
//        vgg:add(MaxPooling(2,2,2,2):ceil())
//
//        ConvBNReLU(128,256):add(nn.Dropout(0.4))
//        ConvBNReLU(256,256):add(nn.Dropout(0.4))
//        ConvBNReLU(256,256)
//        vgg:add(MaxPooling(2,2,2,2):ceil())
//
//        ConvBNReLU(256,512):add(nn.Dropout(0.4))
//        ConvBNReLU(512,512):add(nn.Dropout(0.4))
//        ConvBNReLU(512,512)
//        vgg:add(MaxPooling(2,2,2,2):ceil())
//
//        ConvBNReLU(512,512):add(nn.Dropout(0.4))
//        ConvBNReLU(512,512):add(nn.Dropout(0.4))
//        ConvBNReLU(512,512)
//        vgg:add(MaxPooling(2,2,2,2):ceil())
//        vgg:add(nn.View(512))
//
//        classifier = nn.Sequential()
//        classifier:add(nn.Dropout(0.5))
//        classifier:add(nn.Linear(512,512))
//        classifier:add(nn.BatchNormalization(512))
//        classifier:add(nn.ReLU(true))
//        classifier:add(nn.Dropout(0.5))
//        classifier:add(nn.Linear(512,nClasses))
//        classifier:add(nn.LogSoftMax())
//        vgg:add(classifier)
//        local model = vgg
//
//        local parameters, gradParameters = model:getParameters()
//        model:zeroGradParameters()
//        parameters_initial = parameters : clone()
//        gradParameters_initial = gradParameters : clone()
//
//        criterion =  nn.ClassNLLCriterion()
//
//        state = {
//          learningRate = 1e-2,
//          momentum = 0.9,
//          dampening = 0.0,
//          weightDecay = 5e-4
//        }
//
//        feval = function(x)
//          model:zeroGradParameters()
//          model_initial = model : clone()
//
//          local output1 = model:forward(input)
//          local err1 = criterion:forward(output1, labels)
//          local gradOutput1 = criterion:backward(output1, labels)
//          model:backward(input, gradOutput1)
//          return err1, gradParameters
//        end
//
//        for i = 1,5,1 do
//          w, err = optim.sgd(feval, parameters, state)
//        end
//
//        output=model.output
//        gradOutput=criterion.gradInput
//        gradInput = model.gradInput
//
//      """
//
//    val th = new TH
//    val (luaTime, torchResult) = th.run(code, Map("input"->input, "labels"->labels), Array("output", "gradOutput","err","parameters_initial","gradParameters_initial", "gradInput", "parameters"))
//    val outputTorch = torchResult("output").asInstanceOf[Tensor[Double]]
//    val errTorch = torchResult("err").asInstanceOf[HashMap[Double, Double]].get(1.0)
//    val parameterTorch = torchResult("parameters_initial").asInstanceOf[Tensor[Double]]
//    val gradparameterTorch = torchResult("gradParameters_initial").asInstanceOf[Tensor[Double]]
//    val gradInputTorch = torchResult("gradInput").asInstanceOf[Tensor[Double]]
//    val gradOutputTorch = torchResult("gradOutput").asInstanceOf[Tensor[Double]]
//
//    require(parameters == parameterTorch , "parameter compare failed")
//
//    require(gradparameters == gradparameterTorch, "gradparameter compare failed")
//
//    val (weights, grad) = model.getParameters()
//    val criterion = new ClassNLLCriterion[Double]()
//
//    val state = T("learningRate" -> 1e-2, "momentum" -> 0.9, "weightDecay" -> 5e-4, "dampening" -> 0.0)
//
//    val sgd = new SGD[Double]
//
//    val epsilon = System.getProperty("DoubleTensorEpsilon", "0.0000001").toDouble
//
//    for(i <- 1 to 4) {
//      model.zeroGradParameters()
//      val outputtest = model.forward(input)
//      val loss = criterion.forward(outputtest, labels)
//      val gradoutputtest = criterion.backward(outputtest, labels)
//      model.backward(input, gradoutputtest)
//      sgd.optimize(_ => (loss, grad), weights, state, state)
//    }
//
//    model.zeroGradParameters()
//    val outputTest = model.forward(input)
//    val errTest = criterion.forward(outputTest, labels)
//    val gradOutputTest = criterion.backward(outputTest, labels)
//    val gradInput = model.backward(input, gradOutputTest)
//    sgd.optimize(_ => (errTest, grad), weights, state, state)
//
//    var j = 0
//    var gradInputAbs = 0.0
//    while(j < gradInput.nElement()){
//      val tmp = abs(gradInput.storage().array()(j) - gradInputTorch.storage().array()(j))
//      gradInputAbs += tmp
//      //      if (tmp > 1e-10) println(s"$j: ${gradInput.storage().array()(j)} ${gradInputTorch.storage().array()(j)}")
//      j += 1
//    }
//    assert(gradInputAbs < 2e-20)
//    println(s"gradInputAbs:$gradInputAbs")
//
//    val parametersTorch = torchResult("parameters").asInstanceOf[Tensor[Double]]
//
//    gradOutputTest should be (gradOutputTorch)
//
//    var parametersAbs = 0.0
//    while(j < parameters.nElement()){
//      val tmp = abs(parameters.storage().array()(j) - parametersTorch.storage().array()(j))
//      parametersAbs += tmp
//      //      if (tmp > 1e-10) println(s"$j: ${gradInput.storage().array()(j)} ${gradInputTorch.storage().array()(j)}")
//      j += 1
//    }
//    assert(parametersAbs < 2e-15)
//    println(s"parametersAbs:$parametersAbs")
//
//    var outputAbs = 0.0
//    outputTest.map(outputTorch, (v1, v2) => {
//      outputAbs += abs(v1 - v2)
//      //      if(v1 != v2) println(s"$v1 $v2")
//      //      assert(abs(v1 - v2) < 1e-14);
//      v1
//    })
//    assert(outputAbs < 2e-14)
//    println(s"outputAbs:$outputAbs")
//
//    println(s"err:${abs(errTest - errTorch)}")
//    assert(abs(errTest - errTorch) == 0)
//
//  }
//
//  "VggBn evaluate" should "generate correct output" in {
//    Random.setSeed(3)
//    val input = torch.Tensor[Double](4, 3, 32, 32).apply1(e => Random.nextDouble())
//    val labels = torch.Tensor[Double](4).apply1(e => Random.nextInt(10))
//
//    val seed = 100
//    RNG.setSeed(seed)
//
//    val code =  "torch.manualSeed(" + seed +")\n" +
//      """
//        local nClasses = 10
//        local vgg = nn.Sequential()
//
//        -- building block
//        local function ConvBNReLU(nInputPlane, nOutputPlane)
//          vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
//          vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
//          vgg:add(nn.ReLU(true))
//          return vgg
//        end
//
//        -- Will use "ceil" MaxPooling because we want to save as much
//        -- space as we can
//        local MaxPooling = nn.SpatialMaxPooling
//
//        ConvBNReLU(3,64)
//        ConvBNReLU(64,64)
//        vgg:add(MaxPooling(2,2,2,2):ceil())
//
//        ConvBNReLU(64,128)
//        ConvBNReLU(128,128)
//        vgg:add(MaxPooling(2,2,2,2):ceil())
//
//        ConvBNReLU(128,256)
//        ConvBNReLU(256,256)
//        ConvBNReLU(256,256)
//        vgg:add(MaxPooling(2,2,2,2):ceil())
//
//        ConvBNReLU(256,512)
//        ConvBNReLU(512,512)
//        ConvBNReLU(512,512)
//        vgg:add(MaxPooling(2,2,2,2):ceil())
//
//        ConvBNReLU(512,512)
//        ConvBNReLU(512,512)
//        ConvBNReLU(512,512)
//        vgg:add(MaxPooling(2,2,2,2):ceil())
//        vgg:add(nn.View(512))
//
//        classifier = nn.Sequential()
//        classifier:add(nn.Linear(512,512))
//        classifier:add(nn.BatchNormalization(512))
//        classifier:add(nn.ReLU(true))
//        classifier:add(nn.Linear(512,nClasses))
//        classifier:add(nn.LogSoftMax())
//        vgg:add(classifier)
//        local model = vgg
//
//        local parameters, gradParameters = model:getParameters()
//        model:zeroGradParameters()
//        parameters_initial = parameters : clone()
//        gradParameters_initial = gradParameters : clone()
//
//        criterion =  nn.ClassNLLCriterion()
//
//        state = {
//          learningRate = 1e-2,
//          momentum = 0.9,
//          dampening = 0.0,
//          weightDecay = 5e-4
//        }
//
//        feval = function(x)
//          model:zeroGradParameters()
//          model_initial = model : clone()
//
//          local output1 = model:forward(input)
//          local err1 = criterion:forward(output1, labels)
//          local gradOutput1 = criterion:backward(output1, labels)
//          model:backward(input, gradOutput1)
//          return err1, gradParameters
//        end
//
//        for i = 1,20,1 do
//          w, err = optim.sgd(feval, parameters, state)
//        end
//      """
//    val th = new TH
//    val (luaTime, torchResult) = th.run(code, Map("input"->input, "labels"->labels), Array("model"))
//
//    val model = torchResult("model").asInstanceOf[Module[Double]]
//    val code2 ="""
//        model2=torch.load("/tmp/model.t7")
//        model2:evaluate()
//        output = model2:forward(input)
//      """
//    val (luaTime2, torchResult2) = th.run(code2, Map("input"->input), Array("output", "model2"))
//    val modelTorch = torchResult2("model2").asInstanceOf[Module[Double]]
//    val outputTorch = torchResult2("output").asInstanceOf[Tensor[Double]]
//
//    val (weights, grad) = model.getParameters()
//    println(s"model size: ${weights.nElement()}")
//
//    model.zeroGradParameters()
//    model.evaluate()
//    val outputTest = model.forward(input)
//
//
//    var outputAbs = 0.0
//    outputTest.map(outputTorch, (v1, v2) => {
//      outputAbs += abs(v1 - v2)
//      //            if(v1 != v2) println(s"$v1 $v2")
//      //      assert(abs(v1 - v2) < 1e-14);
//      v1
//    })
//    //    assert(outputAbs < 2e-14)
//    println(s"outputAbs:$outputAbs")
//
////    for(i <- 0 until model.modules.size){
////      println(i)
////      for(j <- 0 until model.modules(i).output.nElement()){
////        if(model.modules(i).output.storage().array()(j) != modelTorch.modules(i).output.storage().array()(j)){
////          println(s"$i $j ${model.modules(i).output.storage().array()(j)} ${modelTorch.modules(i).output.storage().array()(j)}")
////        }
////      }
////    }
//
////    for(i <- 0 until model.modules(45).modules.size){
////      println(i)
////      for(j <- 0 until model.modules(45).modules(i).output.nElement()){
////        if(model.modules(45).modules(i).output.storage().array()(j) != modelTorch.modules(45).modules(i).output.storage().array()(j)){
////          println(s"$i $j ${model.modules(45).modules(i).output.storage().array()(j)} ${modelTorch.modules(45).modules(i).output.storage().array()(j)}")
////        }
////      }
////    }
////    for(j <- 0 until model.modules(45).modules(3).output.nElement()) {
////      if (model.modules(45).modules(3).output.storage().array()(j) != modelTorch.modules(45).modules(3).output.storage().array()(j)) {
////        println(s"3 $j ${model.modules(45).modules(3).output.storage().array()(j)} ${modelTorch.modules(45).modules(3).output.storage().array()(j)}")
////      }
////    }
////    println(s"err:${abs(errTest - errTorch)}")
////    assert(abs(errTest - errTorch) == 0)
//
//  }
//
//  "VggBnDrop evaluate" should "generate correct output" in {
//    Random.setSeed(3)
//    val input = torch.Tensor[Double](4, 3, 32, 32).apply1(e => Random.nextDouble())
//    val labels = torch.Tensor[Double](4).apply1(e => Random.nextInt(10))
//
//    val seed = 100
//
//    val code =
//      """
//        local nClasses = 10
//        local vgg = nn.Sequential()
//
//        -- building block
//        local function ConvBNReLU(nInputPlane, nOutputPlane)
//          vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
//          vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
//          vgg:add(nn.ReLU(true))
//          return vgg
//        end
//
//        -- Will use "ceil" MaxPooling because we want to save as much
//        -- space as we can
//        local MaxPooling = nn.SpatialMaxPooling
//
//        ConvBNReLU(3,64):add(nn.Dropout(0.3))
//        ConvBNReLU(64,64)
//        vgg:add(MaxPooling(2,2,2,2):ceil())
//
//        ConvBNReLU(64,128):add(nn.Dropout(0.4))
//        ConvBNReLU(128,128)
//        vgg:add(MaxPooling(2,2,2,2):ceil())
//
//        ConvBNReLU(128,256):add(nn.Dropout(0.4))
//        ConvBNReLU(256,256):add(nn.Dropout(0.4))
//        ConvBNReLU(256,256)
//        vgg:add(MaxPooling(2,2,2,2):ceil())
//
//        ConvBNReLU(256,512):add(nn.Dropout(0.4))
//        ConvBNReLU(512,512):add(nn.Dropout(0.4))
//        ConvBNReLU(512,512)
//        vgg:add(MaxPooling(2,2,2,2):ceil())
//
//        ConvBNReLU(512,512):add(nn.Dropout(0.4))
//        ConvBNReLU(512,512):add(nn.Dropout(0.4))
//        ConvBNReLU(512,512)
//        vgg:add(MaxPooling(2,2,2,2):ceil())
//        vgg:add(nn.View(512))
//
//        classifier = nn.Sequential()
//        classifier:add(nn.Dropout(0.5))
//        classifier:add(nn.Linear(512,512))
//        classifier:add(nn.BatchNormalization(512))
//        classifier:add(nn.ReLU(true))
//        classifier:add(nn.Dropout(0.5))
//        classifier:add(nn.Linear(512,nClasses))
//        classifier:add(nn.LogSoftMax())
//        vgg:add(classifier)
//        local model = vgg
//
//        local parameters, gradParameters = model:getParameters()
//        model:zeroGradParameters()
//        parameters_initial = parameters : clone()
//        gradParameters_initial = gradParameters : clone()
//
//        criterion =  nn.ClassNLLCriterion()
//
//        state = {
//          learningRate = 1e-2,
//          momentum = 0.9,
//          dampening = 0.0,
//          weightDecay = 5e-4
//        }
//
//        feval = function(x)
//          model:zeroGradParameters()
//          model_initial = model : clone()
//
//          local output1 = model:forward(input)
//          local err1 = criterion:forward(output1, labels)
//          local gradOutput1 = criterion:backward(output1, labels)
//          model:backward(input, gradOutput1)
//          return err1, gradParameters
//        end
//
//        for i = 1,20,1 do
//          w, err = optim.sgd(feval, parameters, state)
//        end
//      """
//    val th = new TH
//    val (luaTime, torchResult) = th.run(code, Map("input"->input, "labels"->labels), Array("model"))
//
//    val model = torchResult("model").asInstanceOf[Module[Double]]
//    val code2 =  "torch.manualSeed(" + seed +")\n" + """
//        model2=torch.load("/tmp/model.t7")
//        model2:evaluate()
//        output = model2:forward(input)
//               """
//    val (luaTime2, torchResult2) = th.run(code2, Map("input"->input), Array("output", "model2"))
//    val modelTorch = torchResult2("model2").asInstanceOf[Module[Double]]
//    val outputTorch = torchResult2("output").asInstanceOf[Tensor[Double]]
//
//    val (weights, grad) = model.getParameters()
//    println(s"model size: ${weights.nElement()}")
//
//    model.zeroGradParameters()
//    RNG.setSeed(seed)
//    model.evaluate()
//    val outputTest = model.forward(input)
//
//
//    var outputAbs = 0.0
//    outputTest.map(outputTorch, (v1, v2) => {
//      outputAbs += abs(v1 - v2)
//      //            if(v1 != v2) println(s"$v1 $v2")
//      //      assert(abs(v1 - v2) < 1e-14);
//      v1
//    })
//    //    assert(outputAbs < 2e-14)
//    println(s"outputAbs:$outputAbs")
//
//    //    for(i <- 0 until model.modules.size){
//    //      println(i)
//    //      for(j <- 0 until model.modules(i).output.nElement()){
//    //        if(model.modules(i).output.storage().array()(j) != modelTorch.modules(i).output.storage().array()(j)){
//    //          println(s"$i $j ${model.modules(i).output.storage().array()(j)} ${modelTorch.modules(i).output.storage().array()(j)}")
//    //        }
//    //      }
//    //    }
//
//    //    for(i <- 0 until model.modules(45).modules.size){
//    //      println(i)
//    //      for(j <- 0 until model.modules(45).modules(i).output.nElement()){
//    //        if(model.modules(45).modules(i).output.storage().array()(j) != modelTorch.modules(45).modules(i).output.storage().array()(j)){
//    //          println(s"$i $j ${model.modules(45).modules(i).output.storage().array()(j)} ${modelTorch.modules(45).modules(i).output.storage().array()(j)}")
//    //        }
//    //      }
//    //    }
//    for(j <- 0 until model.modules(53).modules(5).output.nElement()) {
//      if (model.modules(53).modules(5).output.storage().array()(j) != modelTorch.modules(53).modules(5).output.storage().array()(j)) {
//        println(s"3 $j ${model.modules(53).modules(5).output.storage().array()(j)} ${modelTorch.modules(53).modules(5).output.storage().array()(j)}")
//      }
//    }
//    println()
//    //    println(s"err:${abs(errTest - errTorch)}")
//    //    assert(abs(errTest - errTorch) == 0)
//
//  }
}
