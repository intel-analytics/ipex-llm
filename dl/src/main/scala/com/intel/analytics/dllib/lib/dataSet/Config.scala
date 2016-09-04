package com.intel.analytics.dllib.lib.dataSet

import com.intel.analytics.dllib.lib.models.{AlexNet, AlexNet_OWT, GoogleNet_v1, GoogleNet_v2, LeNet5, Lenet_test, Vgg_16, Vgg_19}
import com.intel.analytics.dllib.lib.nn.Module
import com.intel.analytics.dllib.lib.tensor.{Table, Tensor}
import com.intel.analytics.dllib.lib.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
  * Created by zhangli on 16-8-25.
  * contain: all model, all dataLocal all
  */
object Config {
  val regimes = Map(
    "alexnet" -> Array(
      (1, 18, 1e-2, 5e-4),
      (19, 29, 5e-3, 5e-4),
      (30, 43, 1e-3, 0.0),
      (44, 52, 5e-4, 0.0),
      (53, 100000000, 1e-4, 0.0)
    ),
    "googlenet-cf" -> Array(
      (1, 18, 1e-2, 2e-4),
      (19, 29, 5e-3, 2e-4),
      (30, 43, 1e-3, 0.0),
      (44, 52, 5e-4, 0.0),
      (53, 100000000, 1e-4, 0.0)
    ),
    "googlenet_v1" -> Array()
  )
   def getModel [T : ClassTag](classNum : Int)(netType : String)(implicit ev: TensorNumeric[T]): Module[T] = {
     val model = netType match {
       case "alexnet" => AlexNet.apply[T](classNum)
       case "alexnet_owt" => AlexNet_OWT.apply[T](classNum)
       case "googlenet_v1" => GoogleNet_v1.apply[T](classNum)
       //case "googlenet_v2" => GoogleNet_v2.apply[T](classNum)
       case "LeNet5" => LeNet5.apply[T](classNum)
       case "Vgg_16" => Vgg_16.apply[T](classNum)
       case "Vgg_19" => Vgg_19.apply[T](classNum)
       case _ => ???
     }
     model
   }

  def getDataSet(dataSetType : String, params : Utils.Params)={
    var dataMode = dataSetType match {
      case "MNIST" => new MNISTType(params)
      case "ImageNet" => new ImageNetType(params)
      case "Cifar-10" => new CifarType(params)
      case _ => new MNISTType(params)
    }
    dataMode
  }

  class MNISTType(params : Utils.Params) extends dataType {
    //contain channel, read method, support model,
    channelNum = 1
    supportModels = Array("alexnet","LeNet5")
    loadMethod = "loadBinaryFile"
    meanStd = "grayMeanStd"
    Tensor = "toTensorForGray"
    featureShape = Array(1, 28, 28)
  }
  class ImageNetType(params : Utils.Params) extends dataType {
    channelNum = 3
    supportModels = Array("alexnet","googlenet_v1")
    loadMethod = "loadPrimaryImage"
    meanStd = "rgbMeanStd"
    Tensor = "toTensorRGB"
    featureShape = Array(3, 224, 224)
  }
  class CifarType(params : Utils.Params) extends dataType{
    channelNum = 3
    supportModels = Array("alexnet","LeNet5")
    loadMethod = "loadPrimaryImage"
    meanStd = "rgbMeanStd"
    Tensor = "toTensorRGB"
    featureShape = Array(3, 32, 32)
  }

  trait dataType {
    var channelNum = 3
    var supportModels = new Array[String](1000)
    var loadMethod = ""
    var meanStd = ""
    var Tensor = ""
    var featureShape = new Array[Int](3)
  }

}



