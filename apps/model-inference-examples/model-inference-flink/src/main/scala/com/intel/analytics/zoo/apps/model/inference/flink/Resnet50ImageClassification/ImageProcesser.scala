package com.intel.analytics.zoo.apps.model.inference.flink.Resnet50ImageClassification

class ImageProcessor extends ImageProcessing {
  def preProcess(bytes: Array[Byte], cropWidth: Int, cropHeight: Int, meanR: Int, meanG: Int, meanB: Int, scale: Double) = {
    val imageMat = byteArrayToMat(bytes)
    val imageCent = centerCrop(imageMat, cropWidth, cropHeight)
    val imageArray = matToNCHWAndArray(imageCent)
    imageArray
  }
}
