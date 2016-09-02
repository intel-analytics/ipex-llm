package com.intel.webscaleml.nn.nn.th

import com.intel.webscaleml.nn.nn._
import com.intel.webscaleml.nn.tensor.{Tensor, torch}
import com.intel.webscaleml.nn.th.TH
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import com.intel.webscaleml.nn.tensor.RandomGenerator._

import scala.math._
import scala.sys.process._

/**
  * Created by leon on 16-1-20.
  */
class ConcatSpec extends FlatSpec with BeforeAndAfter with Matchers{
  before{
    if(TH.hasTorch()){
      cancel("Torch is not installed")
    }
  }

  "A Concat Container with Linear" should "generate correct output and grad " in {
    val seed = 2
    RNG.setSeed(seed)
    val module = new Concat[Double](2)
    val layer1 = new Sequential[Double]()
    val layer2 = new Sequential[Double]()
    layer1.add(new SpatialBatchNormalization[Double](3, 1e-3))
    layer2.add(new SpatialBatchNormalization[Double](3, 1e-3))
    module.add(layer1).add(layer2)

    val input = torch.Tensor[Double](4, 3, 4, 4).randn()
    val gradOutput = torch.Tensor[Double](4, 6, 4, 4).randn()

    val code =  "torch.manualSeed(" + seed +")\n" +
             """
    module = nn.Concat(2)
    layer1 = nn.Sequential()
    layer2 = nn.Sequential()
    layer1:add(nn.SpatialBatchNormalization(3, 1e-3))
    layer2:add(nn.SpatialBatchNormalization(3, 1e-3))
    module:add(layer1):add(layer2)
    local parameters, gradParameters = module:getParameters()
    module:zeroGradParameters()
    parameters_initial = parameters : clone()
    gradParameters_initial = gradParameters : clone()

    output = module:forward(input)
    gradInput = module:backward(input,gradOutput)
               """

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput), Array("output", "gradInput", "module", "parameters_initial", "gradParameters_initial"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val gradParametersInitial = torchResult("gradParameters_initial").asInstanceOf[Tensor[Double]]
    val parametersInitial = torchResult("parameters_initial").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaModule = torchResult("module").asInstanceOf[Module[Double]]

    val (parameters, gradParameters) = module.getParameters()
    require(gradParametersInitial == gradParameters)
    require(parametersInitial == parameters)

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input,gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    luaOutput.map(output, (v1, v2) => {
      assert(abs(v1 - v2) == 0)
      v1
    })
    luaGradInput.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) == 0)
      v1
    })

    println("Test case : Concat, Torch : " + luaTime + " s, Scala : " + scalaTime/1e9.toDouble + " s")
  }

  "A Concat Container" should "generate correct output and grad" in {
    val module = new Concat[Double](2)
    val layer1 = new Sequential[Double]()
    val layer2 = new Sequential[Double]()
    layer1.add(new LogSoftMax())
    layer2.add(new LogSoftMax())
    module.add(layer1).add(layer2)

    val input = torch.Tensor[Double](4, 1000).randn()
    val gradOutput = torch.Tensor[Double](4, 2000).randn()

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input,gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = """
    module = nn.Concat(2)
    layer1 = nn.Sequential()
    layer2 = nn.Sequential()
    layer1:add(nn.LogSoftMax())
    layer2:add(nn.LogSoftMax())
    module:add(layer1):add(layer2)
    module:zeroGradParameters()

    output = module:forward(input)
    gradInput = module:backward(input,gradOutput)
               """

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput), Array("output", "gradInput", "module"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaModule = torchResult("module").asInstanceOf[Module[Double]]

    luaOutput should be (output)
    luaGradInput should be (gradInput)

    println("Test case : Concat, Torch : " + luaTime + " s, Scala : " + scalaTime/1e9.toDouble + " s")
  }
}
