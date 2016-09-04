package com.intel.analytics.dllib.lib.tensor

import com.intel.analytics.dllib.lib.nn._
import org.scalatest.{Matchers, FlatSpec}

class FileSpec extends FlatSpec with Matchers {
  "save/load Java object file" should "work properly" in {

    val tmpFile = java.io.File.createTempFile("module", "obj")
    val absolutePath = tmpFile.getAbsolutePath


    val module = new Sequential[Double]

    module.add(new SpatialConvolution(1, 6, 5, 5))
    module.add(new Tanh())
    module.add(new SpatialMaxPooling(2, 2, 2, 2))
    // stage 2 : filter bank -> squashing -> max pooling
    module.add(new SpatialConvolution(6, 12, 5, 5))
    module.add(new Tanh())
    module.add(new SpatialMaxPooling(2, 2, 2, 2))
    //      -- stage 3 : standard 2-layer neural network
    module.add(new Reshape(Array(12 * 5 * 5)))
    module.add(new Linear(12 * 5 * 5, 100))
    module.add(new Tanh())
    module.add(new Linear(100, 6))
    module.add(new LogSoftMax[Double]())

    torch.saveObj(module, absolutePath, true)
    val testModule : Module[Double] = torch.loadObj(absolutePath)

    testModule should be(module)
  }

}
