/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.integration.torch.models

import com.intel.analytics.bigdl.models.inception._
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Graph, Input}
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.optim.SGD
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.integration.torch.{TH, TorchSpec}
import com.intel.analytics.bigdl.models.Inception
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.bigdl.numeric.NumericFloat

import scala.math._
import scala.util.Random

class InceptionSpec extends TorchSpec {
  "Inception+bn" should "generate correct output" in {
    torchCheck()

    Random.setSeed(4)
    val input = Tensor[Double](4, 3, 224, 224).apply1(e => Random.nextDouble())
    val labels = Tensor[Double](4).apply1(e => Random.nextInt(1000))

    val seed = 890
    RNG.setSeed(seed)

    val code = "torch.manualSeed(" + seed + ")\n" +
      """
        local nClasses = 1000
        local function inception(input_size, config)
          local concat = nn.Concat(2)
          if config[1][1] ~= 0 then
             local conv1 = nn.Sequential()
             conv1:add(nn.SpatialConvolution(input_size, config[1][1],1,1,1,1))
             conv1:add(nn.SpatialBatchNormalization(config[1][1],1e-3))
             conv1:add(nn.ReLU(true))
             concat:add(conv1)
          end
          local conv3 = nn.Sequential()
          conv3:add(nn.SpatialConvolution(input_size, config[2][1],1,1,1,1))
          conv3:add(nn.SpatialBatchNormalization(config[2][1],1e-3))
          conv3:add(nn.ReLU(true))
          conv3:add(nn.SpatialConvolution(config[2][1], config[2][2],3,3,1,1,1,1))
          conv3:add(nn.SpatialBatchNormalization(config[2][2],1e-3))
          conv3:add(nn.ReLU(true))
          concat:add(conv3)
          local conv3xx = nn.Sequential()
          conv3xx:add(nn.SpatialConvolution(  input_size, config[3][1],1,1,1,1))
          conv3xx:add(nn.SpatialBatchNormalization(config[3][1],1e-3))
          conv3xx:add(nn.ReLU(true))
          conv3xx:add(nn.SpatialConvolution(config[3][1], config[3][2],3,3,1,1,1,1))
          conv3xx:add(nn.SpatialBatchNormalization(config[3][2],1e-3))
          conv3xx:add(nn.ReLU(true))
          conv3xx:add(nn.SpatialConvolution(config[3][2], config[3][2],3,3,1,1,1,1))
          conv3xx:add(nn.SpatialBatchNormalization(config[3][2],1e-3))
          conv3xx:add(nn.ReLU(true))
          concat:add(conv3xx)
          local pool = nn.Sequential()
          pool:add(nn.SpatialZeroPadding(1,1,1,1)) -- remove after getting nn R2 into fbcode
          if config[4][1] == 'max' then
             pool:add(nn.SpatialMaxPooling(3,3,1,1):ceil())
          elseif config[4][1] == 'avg' then
             pool:add(nn.SpatialAveragePooling(3,3,1,1):ceil())
          else
             error('Unknown pooling')
          end
          if config[4][2] ~= 0 then
             pool:add(nn.SpatialConvolution(input_size, config[4][2],1,1,1,1))
             pool:add(nn.SpatialBatchNormalization(config[4][2],1e-3))
             pool:add(nn.ReLU(true))
          end
          concat:add(pool)
          return concat
        end
        local features = nn.Sequential()
        features:add(nn.SpatialConvolution(3,64,7,7,2,2,3,3))
        features:add(nn.SpatialBatchNormalization(64,1e-3))
        features:add(nn.ReLU(true))
        features:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
        features:add(nn.SpatialConvolution(64,64,1,1)):add(nn.ReLU(true))
        features:add(nn.SpatialConvolution(64,192,3,3,1,1,1,1))
        features:add(nn.SpatialBatchNormalization(192,1e-3))
        features:add(nn.ReLU(true))
        features:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
        features:add(inception( 192, {{ 64},{ 64, 64},{ 64, 96},{'avg', 32}})) -- 3(a)
        features:add(inception( 256, {{ 64},{ 64, 96},{ 64, 96},{'avg', 64}})) -- 3(b)
        features:add(inception( 320, {{  0},{128,160},{ 64, 96},{'max',  0}})) -- 3(c)
        features:add(nn.SpatialConvolution(576,576,2,2,2,2))
        features:add(inception( 576, {{224},{ 64, 96},{ 96,128},{'avg',128}})) -- 4(a)
        features:add(inception( 576, {{192},{ 96,128},{ 96,128},{'avg',128}})) -- 4(b)
        features:add(inception( 576, {{160},{128,160},{128,160},{'avg', 96}})) -- 4(c)
        features:add(inception( 576, {{ 96},{128,192},{160,192},{'avg', 96}})) -- 4(d)
        local main_branch = nn.Sequential()
        main_branch:add(inception( 576, {{  0},{128,192},{192,256},{'max',  0}})) -- 4(e)
        main_branch:add(nn.SpatialConvolution(1024,1024,2,2,2,2))
        main_branch:add(nn.SpatialBatchNormalization(1024,1e-3))
        main_branch:add(inception(1024, {{352},{192,320},{160,224},{'avg',128}})) -- 5(a)
        main_branch:add(inception(1024, {{352},{192,320},{192,224},{'max',128}})) -- 5(b)
        main_branch:add(nn.SpatialAveragePooling(7,7,1,1))
        main_branch:add(nn.View(1024):setNumInputDims(3))
        main_branch:add(nn.Linear(1024,nClasses))
        main_branch:add(nn.LogSoftMax())
        -- add auxillary classifier here (thanks to Christian Szegedy for the details)
        local aux_classifier = nn.Sequential()
        aux_classifier:add(nn.SpatialAveragePooling(5,5,3,3):ceil())
        aux_classifier:add(nn.SpatialConvolution(576,128,1,1,1,1))
        aux_classifier:add(nn.SpatialBatchNormalization(128,1e-3))
        aux_classifier:add(nn.View(128*4*4):setNumInputDims(3))
        aux_classifier:add(nn.Linear(128*4*4,768))
        aux_classifier:add(nn.ReLU(true))
        aux_classifier:add(nn.Linear(768,nClasses))
        aux_classifier:add(nn.LogSoftMax())
        local splitter = nn.Concat(2)
        splitter:add(main_branch):add(aux_classifier)
        local model = nn.Sequential():add(features):add(splitter)
        parameters, gradParameters = model:getParameters()
        model:zeroGradParameters()
        parameters_initial = parameters : clone()
        gradParameters_initial = gradParameters : clone()
        criterion =  nn.ClassNLLCriterion()
        state = {
          learningRate = 1e-2,
          momentum = 0.9,
          dampening = 0.0,
          weightDecay = 5e-4
        }
        feval = function(x)
          model:zeroGradParameters()
          model_initial = model : clone()
          local output1 = model:forward(input)
          local err1 = criterion:forward(output1, labels)
          local gradOutput1 = criterion:backward(output1, labels)
          model:backward(input, gradOutput1)
          return err1, gradParameters
        end
        w, err = optim.sgd(feval, parameters, state)
        output=model.output
        gradOutput=criterion.gradInput
        gradInput = model.gradInput
        model2=model:get(2)
        parameters, gradParameters = model:getParameters()
      """

    TH.runNM(code, Map("input" -> input, "labels" -> labels), Array("output", "gradOutput", "err",
      "parameters_initial", "gradParameters_initial", "gradParameters", "parameters", "model2"),
      suffix)

    val model = Inception.getModel[Double](1000, "inception-bn")

    val parameters = model.getParameters()._1.asInstanceOf[Tensor[Double]]
    println(s"model size: ${parameters.nElement()}")
    val parametersInitTorch = TH.map("parameters_initial", suffix).asInstanceOf[Tensor[Double]]
    require(parameters == parametersInitTorch, "parameter compare failed")

    val gradGarametersInitTorch = TH.map("gradParameters_initial", suffix)
      .asInstanceOf[Tensor[Double]]
    val gradparameters = model.getParameters()._2.asInstanceOf[Tensor[Double]]
    require(gradparameters == gradGarametersInitTorch, "gradparameter compare failed")

    val (weights, grad) = model.getParameters()
    val criterion = new ClassNLLCriterion[Double]()
    val sgd = new SGD[Double]
    val state = T("learningRate" -> 1e-2, "momentum" -> 0.9, "weightDecay" -> 5e-4,
      "dampening" -> 0.0)

    model.zeroGradParameters()
    val outputTest = model.forward(input).toTensor[Double]
    val outputTorch = TH.map("output", suffix).asInstanceOf[Tensor[Double]]
    outputTest shouldEqual outputTorch

    val errTorch = TH.map("err", suffix).asInstanceOf[Table][Double](1)
    val errTest = criterion.forward(outputTest, labels)
    println(s"err:${abs(errTest - errTorch)}")
    assert(abs(errTest - errTorch) < 4e-15)

    val gradOutputTorch = TH.map("gradOutput", suffix).asInstanceOf[Tensor[Double]]
    val gradOutputTest = criterion.backward(outputTest, labels)
    model.backward(input, gradOutputTest)
    gradOutputTest shouldEqual gradOutputTorch

    sgd.optimize(_ => (errTest, grad), weights, state, state)

    val gradParametersTorch = TH.map("gradParameters", suffix).asInstanceOf[Tensor[Double]]
    grad.equals(gradParametersTorch) should be (true)
    val parametersTorch = TH.map("parameters", suffix).asInstanceOf[Tensor[Double]]
    parameters.equals(parametersTorch) should be (true)
  }

  "Inception" should "generate correct output" in {
    torchCheck()

    Random.setSeed(3)
    val input = Tensor[Double](4, 3, 224, 224).apply1(e => Random.nextDouble())
    val labels = Tensor[Double](4).apply1(e => Random.nextInt(1000))

    val seed = 100
    RNG.setSeed(seed)
    val model = Inception.getModel[Double](1000, "inception")

    val code = "torch.manualSeed(" + seed + ")\n" +
      """
        local nClasses = 1000
        local function inception(input_size, config)
          local concat = nn.Concat(2)
          if config[1][1] ~= 0 then
             local conv1 = nn.Sequential()
             conv1:add(nn.SpatialConvolution(input_size, config[1][1],1,1,1,1))
             conv1:add(nn.ReLU(true))
             concat:add(conv1)
          end

          local conv3 = nn.Sequential()
          conv3:add(nn.SpatialConvolution(  input_size, config[2][1],1,1,1,1))
          conv3:add(nn.ReLU(true))

          conv3:add(nn.SpatialConvolution(config[2][1], config[2][2],3,3,1,1,1,1))
          conv3:add(nn.ReLU(true))

          concat:add(conv3)

          local conv3xx = nn.Sequential()
          conv3xx:add(nn.SpatialConvolution(  input_size, config[3][1],1,1,1,1))
          conv3xx:add(nn.ReLU(true))

          conv3xx:add(nn.SpatialConvolution(config[3][1], config[3][2],3,3,1,1,1,1))
          conv3xx:add(nn.ReLU(true))

          conv3xx:add(nn.SpatialConvolution(config[3][2], config[3][2],3,3,1,1,1,1))
          conv3xx:add(nn.ReLU(true))
          concat:add(conv3xx)

          local pool = nn.Sequential()
          pool:add(nn.SpatialZeroPadding(1,1,1,1)) -- remove after getting nn R2 into fbcode
          if config[4][1] == 'max' then
             pool:add(nn.SpatialMaxPooling(3,3,1,1):ceil())
          elseif config[4][1] == 'avg' then
             pool:add(nn.SpatialAveragePooling(3,3,1,1):ceil())
          else
             error('Unknown pooling')
          end
          if config[4][2] ~= 0 then
             pool:add(nn.SpatialConvolution(input_size, config[4][2],1,1,1,1))
             pool:add(nn.ReLU(true))

          end
          concat:add(pool)

          return concat
        end


        local features = nn.Sequential()
        features:add(nn.SpatialConvolution(3,64,7,7,2,2,3,3))
        features:add(nn.ReLU(true))
        features:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
        features:add(nn.SpatialConvolution(64,64,1,1)):add(nn.ReLU(true))
        features:add(nn.SpatialConvolution(64,192,3,3,1,1,1,1))
        features:add(nn.ReLU(true))
        features:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
        features:add(inception( 192, {{ 64},{ 64, 64},{ 64, 96},{'avg', 32}})) -- 3(a)
        features:add(inception( 256, {{ 64},{ 64, 96},{ 64, 96},{'avg', 64}})) -- 3(b)
        features:add(inception( 320, {{  0},{128,160},{ 64, 96},{'max',  0}})) -- 3(c)
        features:add(nn.SpatialConvolution(576,576,2,2,2,2))
        features:add(inception( 576, {{224},{ 64, 96},{ 96,128},{'avg',128}})) -- 4(a)
        features:add(inception( 576, {{192},{ 96,128},{ 96,128},{'avg',128}})) -- 4(b)
        features:add(inception( 576, {{160},{128,160},{128,160},{'avg', 96}})) -- 4(c)
        features:add(inception( 576, {{ 96},{128,192},{160,192},{'avg', 96}})) -- 4(d)

        local main_branch = nn.Sequential()
        main_branch:add(inception( 576, {{  0},{128,192},{192,256},{'max',  0}})) -- 4(e)
        main_branch:add(nn.SpatialConvolution(1024,1024,2,2,2,2))
        --main_branch:add(nn.SpatialBatchNormalization(1024,1e-3))

        main_branch:add(inception(1024, {{352},{192,320},{160,224},{'avg',128}})) -- 5(a)
        main_branch:add(inception(1024, {{352},{192,320},{192,224},{'max',128}})) -- 5(b)
        main_branch:add(nn.SpatialAveragePooling(7,7,1,1))
        main_branch:add(nn.View(1024):setNumInputDims(3))
        main_branch:add(nn.Linear(1024,nClasses))
        main_branch:add(nn.LogSoftMax())

        -- add auxillary classifier here (thanks to Christian Szegedy for the details)
        local aux_classifier = nn.Sequential()
        aux_classifier:add(nn.SpatialAveragePooling(5,5,3,3):ceil())
        aux_classifier:add(nn.SpatialConvolution(576,128,1,1,1,1))
        --aux_classifier:add(nn.SpatialBatchNormalization(128,1e-3))

        aux_classifier:add(nn.View(128*4*4):setNumInputDims(3))
        aux_classifier:add(nn.Linear(128*4*4,768))
        aux_classifier:add(nn.ReLU(true))
        aux_classifier:add(nn.Linear(768,nClasses))
        aux_classifier:add(nn.LogSoftMax())

        local splitter = nn.Concat(2)
        splitter:add(main_branch):add(aux_classifier)
        local model = nn.Sequential():add(features):add(splitter)

        local parameters, gradParameters = model:getParameters()
        model:zeroGradParameters()
        parameters_initial = parameters : clone()
        gradParameters_initial = gradParameters : clone()

        criterion =  nn.ClassNLLCriterion()

        state = {
          learningRate = 1e-2,
          momentum = 0.9,
          dampening = 0.0,
          weightDecay = 5e-4
        }

        feval = function(x)
          model:zeroGradParameters()
          model_initial = model : clone()

          local output1 = model:forward(input)
          local err1 = criterion:forward(output1, labels)
          local gradOutput1 = criterion:backward(output1, labels)
          model:backward(input, gradOutput1)
          return err1, gradParameters
        end

        for i = 1,5,1 do
          w, err = optim.sgd(feval, parameters, state)
        end

        output=model.output
        gradOutput=criterion.gradInput
        gradInput = model.gradInput

      """

    TH.runNM(code, Map("input" -> input, "labels" -> labels), Array("output", "gradOutput", "err",
      "parameters_initial", "gradParameters_initial", "gradInput", "parameters"), suffix)

    val gradInputTorch = TH.map("gradInput", suffix).asInstanceOf[Tensor[Double]]

    val parameters = model.getParameters()._1.asInstanceOf[Tensor[Double]]
    model.zeroGradParameters()
    println(s"model size: ${parameters.nElement()}")
    val parameterTorch = TH.map("parameters_initial", suffix).asInstanceOf[Tensor[Double]]
    require(parameters == parameterTorch, "parameter compare failed")

    val gradparameters = model.getParameters()._2.asInstanceOf[Tensor[Double]]
    val parametersTorch = TH.map("parameters", suffix).asInstanceOf[Tensor[Double]]
    val gradparameterTorch = TH.map("gradParameters_initial", suffix).asInstanceOf[Tensor[Double]]
    require(gradparameters == gradparameterTorch, "gradparameter compare failed")

    val (weights, grad) = model.getParameters()
    val criterion = new ClassNLLCriterion[Double]()

    val state = T("learningRate" -> 1e-2, "momentum" -> 0.9, "weightDecay" -> 5e-4,
      "dampening" -> 0.0)
    val sgd = new SGD[Double]
    val epsilon = System.getProperty("DoubleTensorEpsilon", "0.0000001").toDouble

    for (i <- 1 to 4) {
      model.zeroGradParameters()
      val outputtest = model.forward(input).toTensor[Double]
      val loss = criterion.forward(outputtest, labels)
      val gradoutputtest = criterion.backward(outputtest, labels)
      model.backward(input, gradoutputtest)
      sgd.optimize(_ => (loss, grad), weights, state, state)
    }

    model.zeroGradParameters()
    var outputAbs = 0.0
    val outputTorch = TH.map("output", suffix).asInstanceOf[Tensor[Double]]
    val outputTest = model.forward(input).toTensor[Double]
    outputTest.map(outputTorch, (v1, v2) => {
      outputAbs += abs(v1 - v2)
      v1
    })
    println(s"outputAbs:$outputAbs")

    val errTest = criterion.forward(outputTest, labels)
    val errTorch = TH.map("err", suffix).asInstanceOf[Table][Double](1)
    println(s"err:${abs(errTest - errTorch)}")
    assert(abs(errTest - errTorch) == 0)

    val gradOutputTest = criterion.backward(outputTest, labels)
    val gradOutputTorch = TH.map("gradOutput", suffix).asInstanceOf[Tensor[Double]]
    gradOutputTest shouldEqual gradOutputTorch

    val gradInput = model.backward(input, gradOutputTest)
    gradInput shouldEqual gradInputTorch
    sgd.optimize(_ => (errTest, grad), weights, state, state)
  }

  "load torch's Inception+bn" should "generate correct output" in {
    torchCheck()

    Random.setSeed(4)
    val input = Tensor[Double](4, 3, 224, 224).apply1(e => Random.nextDouble())
    val labels = Tensor[Double](4).apply1(e => Random.nextInt(1000))

    val seed = 890
    RNG.setSeed(seed)

    val code = "torch.manualSeed(" + seed + ")\n" +
      """
        local nClasses = 1000
        local function inception(input_size, config)
          local concat = nn.Concat(2)
          if config[1][1] ~= 0 then
             local conv1 = nn.Sequential()
             conv1:add(nn.SpatialConvolution(input_size, config[1][1],1,1,1,1))
             conv1:add(nn.SpatialBatchNormalization(config[1][1],1e-3))
             conv1:add(nn.ReLU(true))
             concat:add(conv1)
          end
          local conv3 = nn.Sequential()
          conv3:add(nn.SpatialConvolution(input_size, config[2][1],1,1,1,1))
          conv3:add(nn.SpatialBatchNormalization(config[2][1],1e-3))
          conv3:add(nn.ReLU(true))
          conv3:add(nn.SpatialConvolution(config[2][1], config[2][2],3,3,1,1,1,1))
          conv3:add(nn.SpatialBatchNormalization(config[2][2],1e-3))
          conv3:add(nn.ReLU(true))
          concat:add(conv3)
          local conv3xx = nn.Sequential()
          conv3xx:add(nn.SpatialConvolution(  input_size, config[3][1],1,1,1,1))
          conv3xx:add(nn.SpatialBatchNormalization(config[3][1],1e-3))
          conv3xx:add(nn.ReLU(true))
          conv3xx:add(nn.SpatialConvolution(config[3][1], config[3][2],3,3,1,1,1,1))
          conv3xx:add(nn.SpatialBatchNormalization(config[3][2],1e-3))
          conv3xx:add(nn.ReLU(true))
          conv3xx:add(nn.SpatialConvolution(config[3][2], config[3][2],3,3,1,1,1,1))
          conv3xx:add(nn.SpatialBatchNormalization(config[3][2],1e-3))
          conv3xx:add(nn.ReLU(true))
          concat:add(conv3xx)
          local pool = nn.Sequential()
          pool:add(nn.SpatialZeroPadding(1,1,1,1)) -- remove after getting nn R2 into fbcode
          if config[4][1] == 'max' then
             pool:add(nn.SpatialMaxPooling(3,3,1,1):ceil())
          elseif config[4][1] == 'avg' then
             pool:add(nn.SpatialAveragePooling(3,3,1,1):ceil())
          else
             error('Unknown pooling')
          end
          if config[4][2] ~= 0 then
             pool:add(nn.SpatialConvolution(input_size, config[4][2],1,1,1,1))
             pool:add(nn.SpatialBatchNormalization(config[4][2],1e-3))
             pool:add(nn.ReLU(true))
          end
          concat:add(pool)
          return concat
        end
        local features = nn.Sequential()
        features:add(nn.SpatialConvolution(3,64,7,7,2,2,3,3))
        features:add(nn.SpatialBatchNormalization(64,1e-3))
        features:add(nn.ReLU(true))
        features:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
        features:add(nn.SpatialConvolution(64,64,1,1)):add(nn.ReLU(true))
        features:add(nn.SpatialConvolution(64,192,3,3,1,1,1,1))
        features:add(nn.SpatialBatchNormalization(192,1e-3))
        features:add(nn.ReLU(true))
        features:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
        features:add(inception( 192, {{ 64},{ 64, 64},{ 64, 96},{'avg', 32}})) -- 3(a)
        features:add(inception( 256, {{ 64},{ 64, 96},{ 64, 96},{'avg', 64}})) -- 3(b)
        features:add(inception( 320, {{  0},{128,160},{ 64, 96},{'max',  0}})) -- 3(c)
        features:add(nn.SpatialConvolution(576,576,2,2,2,2))
        features:add(inception( 576, {{224},{ 64, 96},{ 96,128},{'avg',128}})) -- 4(a)
        features:add(inception( 576, {{192},{ 96,128},{ 96,128},{'avg',128}})) -- 4(b)
        features:add(inception( 576, {{160},{128,160},{128,160},{'avg', 96}})) -- 4(c)
        features:add(inception( 576, {{ 96},{128,192},{160,192},{'avg', 96}})) -- 4(d)
        local main_branch = nn.Sequential()
        main_branch:add(inception( 576, {{  0},{128,192},{192,256},{'max',  0}})) -- 4(e)
        main_branch:add(nn.SpatialConvolution(1024,1024,2,2,2,2))
        main_branch:add(nn.SpatialBatchNormalization(1024,1e-3))
        main_branch:add(inception(1024, {{352},{192,320},{160,224},{'avg',128}})) -- 5(a)
        main_branch:add(inception(1024, {{352},{192,320},{192,224},{'max',128}})) -- 5(b)
        main_branch:add(nn.SpatialAveragePooling(7,7,1,1))
        main_branch:add(nn.View(1024):setNumInputDims(3))
        main_branch:add(nn.Linear(1024,nClasses))
        main_branch:add(nn.LogSoftMax())
        -- add auxillary classifier here (thanks to Christian Szegedy for the details)
        local aux_classifier = nn.Sequential()
        aux_classifier:add(nn.SpatialAveragePooling(5,5,3,3):ceil())
        aux_classifier:add(nn.SpatialConvolution(576,128,1,1,1,1))
        aux_classifier:add(nn.SpatialBatchNormalization(128,1e-3))
        aux_classifier:add(nn.View(128*4*4):setNumInputDims(3))
        aux_classifier:add(nn.Linear(128*4*4,768))
        aux_classifier:add(nn.ReLU(true))
        aux_classifier:add(nn.Linear(768,nClasses))
        aux_classifier:add(nn.LogSoftMax())
        local splitter = nn.Concat(2)
        splitter:add(main_branch):add(aux_classifier)
        local model = nn.Sequential():add(features):add(splitter)
        local initModel = model:clone()
        parameters, gradParameters = model:getParameters()
        model:zeroGradParameters()
        parameters_initial = parameters : clone()
        gradParameters_initial = gradParameters : clone()
        criterion =  nn.ClassNLLCriterion()
        state = {
          learningRate = 1e-2,
          momentum = 0.9,
          dampening = 0.0,
          weightDecay = 5e-4
        }
        feval = function(x)
          model:zeroGradParameters()
          model_initial = model : clone()
          local output1 = model:forward(input)
          local err1 = criterion:forward(output1, labels)
          local gradOutput1 = criterion:backward(output1, labels)
          model:backward(input, gradOutput1)
          return err1, gradParameters
        end
        w, err = optim.sgd(feval, parameters, state)
        output=model.output
        gradOutput=criterion.gradInput
        gradInput = model.gradInput
        parameters, gradParameters = model:getParameters()
      """

    TH.runNM(code,
      Map("input" -> input, "labels" -> labels),
      Array("output", "gradOutput", "err", "parameters_initial", "gradParameters_initial",
        "gradParameters", "parameters", "initModel"), suffix)

    val model = TH.map("initModel", suffix).
      asInstanceOf[AbstractModule[Tensor[Double], Tensor[Double], Double]]

    val parameters = model.getParameters()._1.asInstanceOf[Tensor[Double]]
    println(s"model size: ${parameters.nElement()}")
    val parametersInitTorch = TH.map("parameters_initial", suffix).asInstanceOf[Tensor[Double]]
    require(parameters == parametersInitTorch, "parameter compare failed")

    val gradGarametersInitTorch = TH.map("gradParameters_initial", suffix)
      .asInstanceOf[Tensor[Double]]
    val gradparameters = model.getParameters()._2.asInstanceOf[Tensor[Double]]
    require(gradparameters == gradGarametersInitTorch, "gradparameter compare failed")

    val (weights, grad) = model.getParameters()
    val criterion = new ClassNLLCriterion[Double]()
    val sgd = new SGD[Double]
    val state = T("learningRate" -> 1e-2, "momentum" -> 0.9, "weightDecay" -> 5e-4,
      "dampening" -> 0.0)

    model.zeroGradParameters()
    val outputTest = model.forward(input)
    val outputTorch = TH.map("output", suffix).asInstanceOf[Tensor[Double]]
    outputTest shouldEqual outputTorch

    val errTorch = TH.map("err", suffix).asInstanceOf[Table][Double](1)
    val errTest = criterion.forward(outputTest, labels)
    println(s"err:${abs(errTest - errTorch)}")
    assert(abs(errTest - errTorch) < 4e-10)

    val gradOutputTorch = TH.map("gradOutput", suffix).asInstanceOf[Tensor[Double]]
    val gradOutputTest = criterion.backward(outputTest, labels)
    model.backward(input, gradOutputTest)
    gradOutputTest shouldEqual gradOutputTorch

    sgd.optimize(_ => (errTest, grad), weights, state, state)
    val gradParametersTorch = TH.map("gradParameters", suffix).asInstanceOf[Tensor[Double]]
    grad == gradParametersTorch should be (true)
    val parametersTorch = TH.map("parameters", suffix).asInstanceOf[Tensor[Double]]
    parameters == parametersTorch should be (true)
  }

  "load torch's Inception+bn float version" should "generate correct output" in {
    torchCheck()

    Random.setSeed(3)
    val input = Tensor[Float](4, 3, 224, 224).apply1(e => Random.nextFloat())
    val labels = Tensor[Float](4).apply1(e => Random.nextInt(1000))

    val seed = 100
    RNG.setSeed(seed)

    val code = "torch.manualSeed(" + seed + ")\n" +
      """
        torch.setdefaulttensortype('torch.FloatTensor')
        local nClasses = 1000
        local function inception(input_size, config)
          local concat = nn.Concat(2)
          if config[1][1] ~= 0 then
             local conv1 = nn.Sequential()
             conv1:add(nn.SpatialConvolution(input_size, config[1][1],1,1,1,1))
             conv1:add(nn.SpatialBatchNormalization(config[1][1],1e-3))
             conv1:add(nn.ReLU(true))
             concat:add(conv1)
          end
          local conv3 = nn.Sequential()
          conv3:add(nn.SpatialConvolution(input_size, config[2][1],1,1,1,1))
          conv3:add(nn.SpatialBatchNormalization(config[2][1],1e-3))
          conv3:add(nn.ReLU(true))
          conv3:add(nn.SpatialConvolution(config[2][1], config[2][2],3,3,1,1,1,1))
          conv3:add(nn.SpatialBatchNormalization(config[2][2],1e-3))
          conv3:add(nn.ReLU(true))
          concat:add(conv3)
          local conv3xx = nn.Sequential()
          conv3xx:add(nn.SpatialConvolution(  input_size, config[3][1],1,1,1,1))
          conv3xx:add(nn.SpatialBatchNormalization(config[3][1],1e-3))
          conv3xx:add(nn.ReLU(true))
          conv3xx:add(nn.SpatialConvolution(config[3][1], config[3][2],3,3,1,1,1,1))
          conv3xx:add(nn.SpatialBatchNormalization(config[3][2],1e-3))
          conv3xx:add(nn.ReLU(true))
          conv3xx:add(nn.SpatialConvolution(config[3][2], config[3][2],3,3,1,1,1,1))
          conv3xx:add(nn.SpatialBatchNormalization(config[3][2],1e-3))
          conv3xx:add(nn.ReLU(true))
          concat:add(conv3xx)
          local pool = nn.Sequential()
          pool:add(nn.SpatialZeroPadding(1,1,1,1)) -- remove after getting nn R2 into fbcode
          if config[4][1] == 'max' then
             pool:add(nn.SpatialMaxPooling(3,3,1,1):ceil())
          elseif config[4][1] == 'avg' then
             pool:add(nn.SpatialAveragePooling(3,3,1,1):ceil())
          else
             error('Unknown pooling')
          end
          if config[4][2] ~= 0 then
             pool:add(nn.SpatialConvolution(input_size, config[4][2],1,1,1,1))
             pool:add(nn.SpatialBatchNormalization(config[4][2],1e-3))
             pool:add(nn.ReLU(true))
          end
          concat:add(pool)
          return concat
        end
        local features = nn.Sequential()
        features:add(nn.SpatialConvolution(3,64,7,7,2,2,3,3))
        features:add(nn.SpatialBatchNormalization(64,1e-3))
        features:add(nn.ReLU(true))
        features:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
        features:add(nn.SpatialConvolution(64,64,1,1)):add(nn.ReLU(true))
        features:add(nn.SpatialConvolution(64,192,3,3,1,1,1,1))
        features:add(nn.SpatialBatchNormalization(192,1e-3))
        features:add(nn.ReLU(true))
        features:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
        features:add(inception( 192, {{ 64},{ 64, 64},{ 64, 96},{'avg', 32}})) -- 3(a)
        features:add(inception( 256, {{ 64},{ 64, 96},{ 64, 96},{'avg', 64}})) -- 3(b)
        features:add(inception( 320, {{  0},{128,160},{ 64, 96},{'max',  0}})) -- 3(c)
        features:add(nn.SpatialConvolution(576,576,2,2,2,2))
        features:add(inception( 576, {{224},{ 64, 96},{ 96,128},{'avg',128}})) -- 4(a)
        features:add(inception( 576, {{192},{ 96,128},{ 96,128},{'avg',128}})) -- 4(b)
        features:add(inception( 576, {{160},{128,160},{128,160},{'avg', 96}})) -- 4(c)
        features:add(inception( 576, {{ 96},{128,192},{160,192},{'avg', 96}})) -- 4(d)
        local main_branch = nn.Sequential()
        main_branch:add(inception( 576, {{  0},{128,192},{192,256},{'max',  0}})) -- 4(e)
        main_branch:add(nn.SpatialConvolution(1024,1024,2,2,2,2))
        main_branch:add(nn.SpatialBatchNormalization(1024,1e-3))
        main_branch:add(inception(1024, {{352},{192,320},{160,224},{'avg',128}})) -- 5(a)
        main_branch:add(inception(1024, {{352},{192,320},{192,224},{'max',128}})) -- 5(b)
        main_branch:add(nn.SpatialAveragePooling(7,7,1,1))
        main_branch:add(nn.View(1024):setNumInputDims(3))
        main_branch:add(nn.Linear(1024,nClasses))
        main_branch:add(nn.LogSoftMax())
        -- add auxillary classifier here (thanks to Christian Szegedy for the details)
        local aux_classifier = nn.Sequential()
        aux_classifier:add(nn.SpatialAveragePooling(5,5,3,3):ceil())
        aux_classifier:add(nn.SpatialConvolution(576,128,1,1,1,1))
        aux_classifier:add(nn.SpatialBatchNormalization(128,1e-3))
        aux_classifier:add(nn.View(128*4*4):setNumInputDims(3))
        aux_classifier:add(nn.Linear(128*4*4,768))
        aux_classifier:add(nn.ReLU(true))
        aux_classifier:add(nn.Linear(768,nClasses))
        aux_classifier:add(nn.LogSoftMax())
        local splitter = nn.Concat(2)
        splitter:add(main_branch):add(aux_classifier)
        local model = nn.Sequential():add(features):add(splitter)
        local initModel = model:clone()
        parameters, gradParameters = model:getParameters()
        model:zeroGradParameters()
        parameters_initial = parameters : clone()
        gradParameters_initial = gradParameters : clone()
        criterion =  nn.ClassNLLCriterion()
        state = {
          learningRate = 1e-2,
          momentum = 0.9,
          dampening = 0.0,
          weightDecay = 5e-4
        }
        feval = function(x)
          model:zeroGradParameters()
          model_initial = model : clone()
          local output1 = model:forward(input)
          local err1 = criterion:forward(output1, labels)
          local gradOutput1 = criterion:backward(output1, labels)
          model:backward(input, gradOutput1)
          return err1, gradParameters
        end
      """

    TH.runNM(code, Map("input" -> input, "labels" -> labels), Array("initModel"), suffix)

    val model = Inception.getModel[Float](1000, "inception-bn")
    val model2 = TH.map("initModel", suffix).
      asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]]
    model2 should be (model)

    val (weights, grad) = model.getParameters()
    val (weights2, grad2) = model2.getParameters()
    // Notice: as a very small different with torch's init parameter, we need to copy the weight.
    weights2.copy(weights)
    val criterion = new ClassNLLCriterion[Float]()
    val sgd = new SGD[Float]
    val state = T("learningRate" -> 1e-2, "momentum" -> 0.9, "weightDecay" -> 5e-4,
      "dampening" -> 0.0)
    val state2 = T("learningRate" -> 1e-2, "momentum" -> 0.9, "weightDecay" -> 5e-4,
      "dampening" -> 0.0)

    for (i <- 1 to 5) {
      model.zeroGradParameters()
      val outputtest = model.forward(input).toTensor[Float]
      val loss = criterion.forward(outputtest, labels)
      val gradoutputtest = criterion.backward(outputtest, labels)
      val gradInput = model.backward(input, gradoutputtest)
      sgd.optimize(_ => (loss, grad), weights, state, state)

      model2.zeroGradParameters()
      val outputtest2 = model2.forward(input)
      val loss2 = criterion.forward(outputtest, labels)
      val gradoutputtest2 = criterion.backward(outputtest, labels)
      val gradInput2 = model2.backward(input, gradoutputtest2)
      sgd.optimize(_ => (loss2, grad2), weights2, state2, state2)
      loss should be (loss2)
      gradInput should be (gradInput2)
      grad.equals(grad2) should be (true)
      outputtest should be (outputtest2)
      gradoutputtest should be (gradoutputtest2)
      weights.equals(weights2) should be (true)
    }
  }

  "Inception ModelCaffe" should "init right" in {
    RNG.setSeed(1024)

    Random.setSeed(1024)

    val input = Tensor[Float](4, 3, 224, 224).apply1(e => Random.nextFloat())
    val labels = Tensor[Float](4).apply1(e => Random.nextInt(1000))

    val model = Inception.getModelCaffe[Float](1000)

    val criterion = new ClassNLLCriterion[Float]()

    model.zeroGradParameters()
    val output = model.forward(input).toTensor[Float]
    val loss = criterion.forward(output, labels)

    // since we already set the seed, the loss should match exactly
    loss should be (6.893043f)
  }

  "InceptionV1 " should "init right" in {
    RNG.setSeed(1024)

    Random.setSeed(1024)

    val input = Tensor[Float](4, 3, 224, 224).apply1(e => Random.nextFloat())
    val labels = Tensor[Float](4).apply1(e => Random.nextInt(1000))

    val model = Inception_v1(1000)

    val criterion = new ClassNLLCriterion[Float]()

    model.zeroGradParameters()
    val output = model.forward(input).toTensor[Float]
    val loss = criterion.forward(output, labels)

    // since we already set the seed, the loss should match exactly
    loss should be (6.6648364f)
  }


}
