/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.transform.vision.image

import java.util.concurrent.atomic.AtomicInteger
import com.intel.analytics.bigdl.dataset.{MiniBatch, Sample, Transformer, Utils}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import com.intel.analytics.bigdl.utils.{Engine, T, Table}
import scala.collection.mutable.IndexedSeq
import scala.reflect.ClassTag

object MTImageFeatureToBatch {
  /**
   * The transformer from ImageFeature to mini-batches
   * @param width width of the output images
   * @param height height of the output images
   * @param batchSize batch size
   * @param transformer pipeline for pre-processing, finally outputting ImageFeature
   * @param toRGB if converted to RGB, default format is BGR
   * @param extractRoi if true, extract ROI labels for segmentation; else the labels are for
   *                   classification
   * @return
   */
  def apply(width: Int, height: Int, batchSize: Int,
            transformer: FeatureTransformer, toRGB: Boolean = true, extractRoi: Boolean = false)
  : MTImageFeatureToBatch = {
    if (extractRoi) {
      new RoiMTImageFeatureToBatch (
        width, height, batchSize, transformer, toRGB)
    } else {
      new ClassificationMTImageFeatureToBatch (
        width, height, batchSize, transformer, toRGB)
    }
  }
}

/**
 * An abstract class to convert ImageFeature iterator to MiniBatches. This transformer will be run
 * on each image feature. "processImageFeature" will be called to buffer the image features. When
 * there are enough buffered image features to form a batch, "createBatch" will be called.
 * You should override processImageFeature to buffer each image feature, and createBatch
 * to convert the buffered data into a mini-batch
 * @param totalBatchSize global batch size
 * @param transformer pipeline for pre-processing
 */
abstract class MTImageFeatureToBatch private[bigdl](
  totalBatchSize: Int, transformer: FeatureTransformer)
  extends Transformer[ImageFeature, MiniBatch[Float]] {

  protected val batchSize: Int = Utils.getBatchSize(totalBatchSize)

  protected val parallelism: Int = Engine.coreNumber()

  private def getPosition(count: AtomicInteger): Int = {
    val position = count.getAndIncrement()
    if (position < batchSize) position else -1
  }

  private lazy val transformers = (1 to parallelism).map(
    _ => new PreFetch -> transformer.cloneTransformer()
  ).toArray

  protected def processImageFeature(img: ImageFeature, position: Int)

  protected def createBatch(batchSize: Int): MiniBatch[Float]

  override def apply(prev: Iterator[ImageFeature]): Iterator[MiniBatch[Float]] = {
    val iterators = transformers.map(_.apply(prev))

    new Iterator[MiniBatch[Float]] {
      override def hasNext: Boolean = {
        iterators.map(_.hasNext).reduce(_ || _)
      }

      override def next(): MiniBatch[Float] = {
        val count = new AtomicInteger(0)
        val batch = Engine.default.invokeAndWait((0 until parallelism).map(tid => () => {
          var position = 0
          var record = 0
          while (iterators(tid).hasNext && {
            position = getPosition(count)
            position != -1
          }) {
            val img = iterators(tid).next()
            processImageFeature(img, position)
            record += 1
          }
          record
        })).sum
        createBatch(batch)
      }
    }
  }
}


private class PreFetch extends Transformer[ImageFeature, ImageFeature] {
  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    new Iterator[ImageFeature] {
      private var buffer: ImageFeature = null.asInstanceOf[ImageFeature]

      override def hasNext: Boolean = {
        if (buffer != null) {
          true
        } else {
          buffer = prev.next()
          if (buffer == null) false else true
        }
      }

      override def next(): ImageFeature = {
        if (buffer == null) {
          prev.next()
        } else {
          val tmp = buffer
          buffer = null.asInstanceOf[ImageFeature]
          tmp
        }
      }
    }
  }
}

/**
 * A transformer pipeline wrapper to create Minibatch in multiple threads for classification
 * @param width final image width
 * @param height final image height
 * @param totalBatchSize global batch size
 * @param transformer pipeline for pre-processing
 * @param toRGB  if converted to RGB, default format is BGR
 */
class ClassificationMTImageFeatureToBatch private[bigdl](width: Int, height: Int,
  totalBatchSize: Int, transformer: FeatureTransformer, toRGB: Boolean = true)
  extends MTImageFeatureToBatch(totalBatchSize, transformer) {

  private val frameLength = height * width
  private val featureData: Array[Float] = new Array[Float](batchSize * frameLength * 3)
  private val labelData: Array[Float] = new Array[Float](batchSize)
  private val featureTensor: Tensor[Float] = Tensor[Float]()
  private val labelTensor: Tensor[Float] = Tensor[Float]()

  override protected def processImageFeature(img: ImageFeature, position: Int): Unit = {
    img.copyTo(featureData, position * frameLength * 3, toRGB = toRGB)
    labelData(position) = img.getLabel.asInstanceOf[Tensor[Float]].valueAt(1)
  }

  override protected def createBatch(batch: Int): MiniBatch[Float] = {
    if (labelTensor.nElement() != batch) {
      featureTensor.set(Storage[Float](featureData),
        storageOffset = 1, sizes = Array(batch, 3, height, width))
      labelTensor.set(Storage[Float](labelData),
        storageOffset = 1, sizes = Array(batch))
    }

    MiniBatch(featureTensor, labelTensor)
  }
}

/**
 * A batch of images with flattened RoiLabels
 * the getTarget() returns a Table with key from 1 to batchSize. Each key in the table is mapped to
 * a Table for the annotation of an image in the batch. The annotation table holds the annotation
 * info for one image (assume the image has N detections). The annotation table has
 *
 * Key                Value
 * RoiLabel.CLASSES   the categories for each detections (see RoiLabel.clasees field)
 *                    (1 x N), or (2 x N) Tensor[Float]
 * RoiLabel.BBOXES    the bboxes, (N x 4) Tensor[Float]
 * RoiLabel.MASKS     (Optional) the mask data, Array[Tensor[Float]\]. The outer array has N
 *                    elements. The inner tensor holds the data for segmentation
 * RoiLabel.ISCROWD   Whether each detection is crowd. (1 x N) Tensor[Float].
 *                    -1: unknown, 0: not crowd, 1: is crowd
 * RoiLabel.ORIGSIZE  The original size of the image, tuple of (height, width, channels)
 */
class RoiMiniBatch(val input: Tensor[Float], val target: IndexedSeq[RoiLabel],
  val isCrowd: IndexedSeq[Tensor[Float]], val originalSizes: IndexedSeq[(Int, Int, Int)])
  extends MiniBatch[Float] {

  override def size(): Int = {
    input.size(1)
  }

  override def getInput(): Tensor[Float] = input

  override def getTarget(): Table = {
    val tables = (target, isCrowd, originalSizes).zipped.map { case (roiLabel, crowd, size) =>
      roiLabel.toTable
        .update(RoiLabel.ISCROWD, crowd)
        .update(RoiLabel.ORIGSIZE, size)
    }
    T.seq(tables)
  }

  override def slice(offset: Int, length: Int): MiniBatch[Float] = {
    val subInput = input.narrow(1, offset, length)
    val subTarget = target.view(offset - 1, length) // offset starts from 1
    val subIsCrowd = isCrowd.view(offset - 1, length) // offset starts from 1
    val subSize = originalSizes.view(offset - 1, length) // offset starts from 1
    RoiMiniBatch(subInput, subTarget, subIsCrowd, subSize)
  }

  override def set(samples: Seq[Sample[Float]])(implicit ev: TensorNumeric[Float])
  : RoiMiniBatch.this.type = {
    throw new NotImplementedError("do not use Sample here")
  }
}

object RoiMiniBatch {
  def apply(data: Tensor[Float], target: IndexedSeq[RoiLabel],
    isCrowd: IndexedSeq[Tensor[Float]], originalSizes: IndexedSeq[(Int, Int, Int)]):
  RoiMiniBatch = new RoiMiniBatch(data, target, isCrowd, originalSizes)
}


/**
 * A transformer pipeline wrapper to create RoiMiniBatch in multiple threads
 * The output "target" is a Table. The keys are from 1 to sizeof(batch). The values are
 * the tables for each RoiLabel. Each Roi label table, contains fields of RoiLabel class.
 * @param width final image width
 * @param height final image height
 * @param totalBatchSize global batch size
 * @param transformer pipeline for pre-processing
 * @param toRGB  if converted to RGB, default format is BGR
 */
class RoiMTImageFeatureToBatch private[bigdl](width: Int, height: Int,
  totalBatchSize: Int, transformer: FeatureTransformer, toRGB: Boolean = true)
  extends MTImageFeatureToBatch(totalBatchSize, transformer) {

  private val frameLength = height * width
  private val featureData: Array[Float] = new Array[Float](batchSize * frameLength * 3)
  private val labelData: Array[RoiLabel] = new Array[RoiLabel](batchSize)
  private val isCrowdData: Array[Tensor[Float]] = new Array[Tensor[Float]](batchSize)
  private val origSizeData: Array[(Int, Int, Int)] = new Array[(Int, Int, Int)](batchSize)
  private var featureTensor: Tensor[Float] = null

  override protected def processImageFeature(img: ImageFeature, position: Int): Unit = {
    img.copyTo(featureData, position * frameLength * 3, toRGB = toRGB)
    val isCrowd = img(RoiLabel.ISCROWD).asInstanceOf[Tensor[Float]]
    val label = img.getLabel.asInstanceOf[RoiLabel]
    require(label.bboxes.size(1) == isCrowd.size(1), "The number of detections" +
      "in ImageFeature's ISCROWD should be equal to the number of detections in the RoiLabel")
    isCrowdData(position) = isCrowd
    labelData(position) = label
    origSizeData(position) = img.getOriginalSize
  }

  override protected def createBatch(batchSize: Int): MiniBatch[Float] = {
    if (featureTensor == null) {
      featureTensor = Tensor(Storage[Float](featureData),
        storageOffset = 1, size = Array(batchSize, 3, height, width))
    }
    RoiMiniBatch(featureTensor, labelData.view, isCrowdData.view, origSizeData.view)
  }
}
