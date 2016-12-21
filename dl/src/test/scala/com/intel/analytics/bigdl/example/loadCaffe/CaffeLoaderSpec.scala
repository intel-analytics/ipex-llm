package com.intel.analytics.bigdl.example.loadCaffe


import com.intel.analytics.bigdl.models.googlenet.GoogleNet_v1_NoAuxClassifier
import com.intel.analytics.bigdl.optim.{Top1Accuracy, Top5Accuracy}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.File
import org.scalatest.{FlatSpec, Matchers}

class CaffeLoaderSpec extends FlatSpec with Matchers {

  val classLoader = getClass().getClassLoader()

  def getPath(name: String): String = {
    classLoader.getResource(s"caffe/$name").getPath
  }

  behavior of "CaffeLoaderSpec"

  it should "load" in {
    val modelPath = getPath("googlenet.caffemodel")
    val defPath = getPath("deploy.prototxt")
    val module = GoogleNet_v1_NoAuxClassifier(1000)

    val model = CaffeLoader.load[Float](module, defPath, modelPath)
    model.evaluate()
    val input = File.load[Tensor[Float]](getPath("data.obj"))
    val target = File.load[Tensor[Float]](getPath("target.obj"))
    val output = model.forward(input)

    def getAccu(resStr: String) = resStr.substring(resStr.lastIndexOf(":") + 1,
      resStr.lastIndexOf(")")).trim.toFloat

    val top1 = getAccu(new Top1Accuracy().apply(output, target).toString())
    val top5 = getAccu(new Top5Accuracy().apply(output, target).toString())

    assert(Math.abs(top1 - 6.600000262260437012e-01) < 1e-6)
    assert(Math.abs(top5 - 8.600000143051147461e-01) < 1e-6)
  }

}
