package com.intel.analytics.bigdl.models.vgg

import java.nio.file.Path

import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.dataset.{CachedDistriDataSet, DistributedDataSet, LocalArrayDataSet, LocalDataSet}
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.spark.SparkContext

object DataSet {
  def localDataSet(imagesFile: Path, looped: Boolean, batchSize : Int)
  : LocalDataSet[(Tensor[Float], Tensor[Float])] = {
    val ds = LocalImageFiles.LocalBytesDataSet(imagesFile, looped, 32)
    val toImage = LabeledBytesToRGBImg()
    val normalizer = LabeledRGBImgNormalizer(ds -> toImage)
    val toTensor = new LabeledRGBImgToTensor(batchSize)
    ds -> toImage -> normalizer -> toTensor
  }

  def distributedDataSet(imagesFile: Path, looped: Boolean, sc: SparkContext,
    partitionNum: Int, batchSize : Int): DistributedDataSet[(Tensor[Float], Tensor[Float])] = {
    val ds = LocalImageFiles.DistriDataSet(imagesFile, looped, sc, partitionNum, 32)

    val toImage = LabeledBytesToRGBImg()
    val normalizer = LabeledRGBImgNormalizer(0.5, 0.5, 0.5, 1.0, 1.0, 1.0)
    val toTensor = new LabeledRGBImgToTensor(batchSize)
    ds -> toImage -> normalizer -> toTensor
  }
}
