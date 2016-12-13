package com.intel.analytics.bigdl.models.vgg

import java.nio.file.Path

import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.spark.SparkContext

object DataSet {
  def localDataSet(imagesFile: Path, looped: Boolean, batchSize : Int)
  : LocalDataSet[Batch[Float]] = {
    val ds = LocalImageFiles.LocalBytesDataSet(imagesFile, looped, 32)
    val toImage = SampleToRGBImg()
    val normalizer = RGBImgNormalizer(ds -> toImage)
    val toBatch = new RGBImgToBatch(batchSize)
    ds -> toImage -> normalizer -> toBatch
  }

  def distributedDataSet(imagesFile: Path, looped: Boolean, sc: SparkContext,
    partitionNum: Int, batchSize : Int): DistributedDataSet[Batch[Float]] = {
    val ds = LocalImageFiles.DistriDataSet(imagesFile, looped, sc, partitionNum, 32)

    val toImage = SampleToRGBImg()
    val normalizer = RGBImgNormalizer(0.5, 0.5, 0.5, 1.0, 1.0, 1.0)
    val toTensor = new RGBImgToBatch(batchSize)
    ds -> toImage -> normalizer -> toTensor
  }
}
