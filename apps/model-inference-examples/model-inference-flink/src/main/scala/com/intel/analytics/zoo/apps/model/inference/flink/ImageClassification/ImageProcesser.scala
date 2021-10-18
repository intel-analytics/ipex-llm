package com.intel.analytics.zoo.apps.model.inference.flink.ImageClassification

class ImageProcessor extends ImageProcessing {
  def preProcess(bytes: Array[Byte], cropWidth: Int, cropHeight: Int ) = {
    // convert Array[byte] to OpenCVMat
    val imageMat = byteArrayToMat(bytes)
    // do a center crop by resizing a square
    val imageCent = centerCrop(imageMat, cropWidth, cropHeight)
    // convert OpenCVMat to Array
    val imageArray = matToArray(imageCent)
    // Normalize with channel and scale
    val imageNorm = channelScaledNormalize(imageArray, 127, 127, 127, 1/127f)
    imageNorm
  }
}
