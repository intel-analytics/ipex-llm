package com.intel.analytics.zoo.apps.model.inference.flink.Resnet50ImageClassification

class ImageProcesser(bytes: Array[Byte], cropWidth: Int, cropHeight: Int) extends ImageProcessing {
  def preProcess(bytes: Array[Byte], cropWidth: Int, cropHeight: Int) = {
    val imageMat = byteArrayToMat(bytes)
    val imageCent = centerCrop(imageMat, cropWidth, cropHeight)
    val imageTensor = matToNCHWAndRGBTensor(imageCent)
    imageTensor
  }
}
