package com.intel.analytics.sparkdl.models

import com.intel.analytics.sparkdl.nn._
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.utils.Table

import scala.reflect.ClassTag

/**
  * Created by ywan on 16-9-23.
  */


object ResNet {
  var iChannels = 0
  def apply[T: ClassTag](classNum: Int, opt: Table)(implicit ev: TensorNumeric[T]): Module[T] = {

    val depth = opt.get("depth").getOrElse(18)
    val shortcutType = if (opt.get("shortcutType") != null) opt.get("shortcutType") else "B"


    def shortcut(nInputPlane: Int, nOutputPlane: Int, stride: Int): Module[T] = {
      val useConv = shortcutType == "C" || (shortcutType == "B" && nInputPlane != nOutputPlane)

      if (useConv == true) {
        val model = new Sequential[T]()
        model.add(new SpatialConvolution[T](nInputPlane, nOutputPlane, 1, 1, stride, stride))
        model.add(new SpatialBatchNormalization(nOutputPlane))
        model
      } else if (nInputPlane != nOutputPlane) {
        val model = new Sequential[T]()
        model.add(new SpatialAveragePooling[T](1, 1, stride, stride))
        model.add(new Concat[T](2)
                  .add(new Identity[T]())
                  .add(new MulConstant[T](ev.fromType(0))))
        model
        // sc.add(new Concat[T](2).add(Identity).add(MulConstant(0)) //  -- ZhangYao's code here
      }  else { val model = new Identity(); model} //    -- ZhangYao's code here
    }

    def basicblock(n: Int, stride: Int): Module[T] = {
      var nInputPlane = iChannels
      iChannels = n

      val s = new Sequential[T]()
      s.add(new SpatialConvolution[T](nInputPlane, n, 3, 3, stride, stride, 1, 1))
      s.add(new SpatialBatchNormalization[T](n))
      s.add(new ReLU[T](true))
      s.add(new SpatialConvolution[T](n ,n, 3, 3, 1, 1, 1, 1))
      s.add(new SpatialBatchNormalization[T](n))

      val model = new Sequential[T]()
      //model.add(new ConcatAddTable[T](true)
      //        .add(s)
      //        .add(shortcut(nInputPlane, n, stride)))
      model.add(s)
      model.add(new ReLU[T](true))
      model
      //val model = new Sequential[T]()
      //model.add(new ConcatTable().add(s).add(shortcut(nInputPlane, n, stride))) //   -- Zhang yao's code here
      //model.add(new CAddTable(true))
      //model.add(new ReLU(true))
      //model
    }

    def bottleneck(n: Int, stride: Int): Module[T] = {
      var nInputPlane   = iChannels
      iChannels = n * 4

      val s = new Sequential[T]()
      s.add(new SpatialConvolution[T](nInputPlane, n, 1, 1, 1, 1, 0, 0))
      s.add(new SpatialBatchNormalization[T](n))
      s.add(new ReLU[T](true))
      s.add(new SpatialConvolution[T](n, n, 3, 3, stride, stride, 1, 1))
      s.add(new SpatialBatchNormalization[T](n))
      s.add(new ReLU[T](true))
      s.add(new SpatialConvolution[T](n, n*4, 1, 1, 1, 1, 0, 0))
      s.add(new SpatialBatchNormalization[T](n * 4))

      val model = new Sequential[T]()
      model.add(new ConcatAddTable[T](true)
        .add(s)
        .add(shortcut(nInputPlane, n*4, stride)))
      model.add(new ReLU[T](true))
      model
      //val model = new Sequential[T]()
      //model.add(new ConcatTable().add(s).add(shortcut(nInputPlane, n*4, stride))) -- Zhang yao's code
      //model.add(new CAddTable(true)) -- Zhangyao's code
      //model.add(new ReLU(true))
    }

    def layer(block: (Int, Int) => Module[T], features: Int, count: Int, stride: Int = 1): Module[T] = {
      val s = new Sequential[T]()
      for (i <- 1 to count) {
        s.add(block(features, if (i == 1) stride else 1))
      }
      s
    }

    val model = new Sequential[T]()
    //if (opt.get("dataset") == "imagenet") {

      // configuration for ResNet-50
      val cfg = Map(
        18  -> ((2, 2, 2, 2), 512, basicblock:(Int, Int) => Module[T]),
        34  -> ((3, 4, 6, 3), 512, basicblock:(Int, Int) => Module[T]),
        50  -> ((3, 4, 6, 3), 2048, bottleneck:(Int, Int) => Module[T]),
        101 -> ((3, 4, 23, 3), 2048, bottleneck:(Int, Int) => Module[T]),
        152 -> ((3, 8, 36, 3), 2048, bottleneck:(Int, Int) => Module[T])
      )

      assert(cfg.keySet.exists(_ == depth))

    val (loopConfig, nFeatures, block) = cfg.get(depth).get
    iChannels = 64
      println(" | ResNet-" + depth + " ImageNet")

      //-- The ResNet ImageNet Model

      model.add(new SpatialConvolution[T](3, 64, 7, 7, 2, 2, 3, 3))
      model.add(new SpatialBatchNormalization[T](64))
      model.add(new ReLU[T](true))
      model.add(new SpatialMaxPooling[T](3, 3, 2, 2, 1, 1))
      model.add(layer(block, 64, loopConfig._1))
      model.add(layer(block, 128, loopConfig._2, 2))
      model.add(layer(block, 256, loopConfig._3, 2))
      model.add(layer(block, 512, loopConfig._4, 2))
      model.add(new SpatialAveragePooling[T](7, 7, 1, 1))
      model.add(new View[T](nFeatures).setNumInputDims(3))
      model.add(new Linear[T](nFeatures, classNum))
      //model.add(new LogSoftMax[T])
    //}
    model

  }
}