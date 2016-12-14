package com.intel.analytics.bigdl.models.vgg

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.numeric.NumericFloat

import scala.reflect.ClassTag

object Vgg {
  def apply(classNum: Int): Module[Float] = {
    val vggBnDo = Sequential[Float]()

    def convBNReLU(nInputPlane: Int, nOutPutPlane: Int)
    : Sequential[Float] = {
      vggBnDo.add(SpatialConvolution(nInputPlane, nOutPutPlane, 3, 3, 1, 1, 1, 1))
      vggBnDo.add(SpatialBatchNormalization(nOutPutPlane, 1e-3))
      vggBnDo.add(ReLU(true))
      vggBnDo
    }
    convBNReLU(3, 64).add(Dropout((0.3)))
    convBNReLU(64, 64)
    vggBnDo.add(SpatialMaxPooling(2, 2, 2, 2).ceil())

    convBNReLU(64, 128).add(Dropout(0.4))
    convBNReLU(128, 128)
    vggBnDo.add(SpatialMaxPooling(2, 2, 2, 2).ceil())

    convBNReLU(128, 256).add(Dropout(0.4))
    convBNReLU(256, 256).add(Dropout(0.4))
    convBNReLU(256, 256)
    vggBnDo.add(SpatialMaxPooling(2, 2, 2, 2).ceil())

    convBNReLU(256, 512).add(Dropout(0.4))
    convBNReLU(512, 512).add(Dropout(0.4))
    convBNReLU(512, 512)
    vggBnDo.add(SpatialMaxPooling(2, 2, 2, 2).ceil())

    convBNReLU(512, 512).add(Dropout(0.4))
    convBNReLU(512, 512).add(Dropout(0.4))
    convBNReLU(512, 512)
    vggBnDo.add(SpatialMaxPooling(2, 2, 2, 2).ceil())
    vggBnDo.add(View(512))

    val classifier = Sequential[Float]()
    classifier.add(Dropout(0.5))
    classifier.add(Linear(512, 512))
    classifier.add(BatchNormalization(512))
    classifier.add(ReLU(true))
    classifier.add(Dropout(0.5))
    classifier.add(Linear(512, classNum))
    classifier.add(LogSoftMax())
    vggBnDo.add(classifier)

    vggBnDo
  }
}