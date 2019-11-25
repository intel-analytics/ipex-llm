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

import com.intel.analytics.bigdl.dataset.segmentation.RLEMasks
import java.util.concurrent.atomic.AtomicInteger
import com.intel.analytics.bigdl.dataset.{MiniBatch, Sample, Transformer, Utils}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import com.intel.analytics.bigdl.utils.{Engine, T, Table}

object MTImageFeatureToBatch {
  /**
   * The transformer from ImageFeature to mini-batches
   * @param width width of the output images
   * @param height height of the output images
   * @param batchSize batch size
   * @param transformer pipeline for pre-processing, finally outputting ImageFeature
   * @param toRGB if converted to RGB, default format is BGR
   * @return
   */
  def apply(width: Int, height: Int, batchSize: Int,
            transformer: FeatureTransformer, toRGB: Boolean = false)
  : MTImageFeatureToBatch = {
      new ClassificationMTImageFeatureToBatch (
        width, height, batchSize, transformer, toRGB)
  }

  private[image] def checkLabels[T](labelData: Array[T]): Array[T] = {
    if (labelData.length == 0) {
      labelData
    } else {
      val hasLabel = labelData.head != null
      for (i <- 1 until labelData.length) {
        val curHasLabel = labelData(i) != null
        require(curHasLabel == hasLabel, "The input data must either be all labeled or" +
          " be all unlabeled")
      }
      if (hasLabel) labelData else null
    }
  }

  private[image] def arraySlice[T](array: Array[T], batchSize: Int) = {
    if (array.length == batchSize) array else array.slice(0, batchSize)
  }
}

import MTImageFeatureToBatch._

object RoiImageFeatureToBatch {
  /**
   * The transformer from ImageFeature to mini-batches, and extract ROI labels for segmentation
   * if roi labels are set. The sizes of the images can be different.
   * @param batchSize global batch size
   * @param transformer pipeline for pre-processing
   * @param toRGB if converted to RGB, default format is BGR
   * @param sizeDivisible when it's greater than 0, height and wide should be divisible by this size
   *
   */
  def withResize(batchSize: Int, transformer: FeatureTransformer,
    toRGB : Boolean = false, sizeDivisible: Int = -1)
  : MTImageFeatureToBatch =
        new RoiImageFeatureToBatchWithResize(sizeDivisible, batchSize, transformer, toRGB)


  /**
   * The transformer from ImageFeature to mini-batches, and extract ROI labels for segmentation
   * if roi labels are set. The sizes of the images must be the same.
   * @param width width of the output images
   * @param height height of the output images
   * @param batchSize global batch size
   * @param transformer pipeline for pre-processing
   * @param toRGB if converted to RGB, default format is BGR
   *
   */
  def apply(width: Int, height: Int, batchSize: Int,
    transformer: FeatureTransformer, toRGB: Boolean = false) : MTImageFeatureToBatch = {
    new RoiImageFeatureToBatch(width, height, batchSize, transformer, toRGB)
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
 * A transformer pipeline wrapper to create labeled Minibatch in multiple threads for classification
 * @param width final image width
 * @param height final image height
 * @param totalBatchSize global batch size
 * @param transformer pipeline for pre-processing
 * @param toRGB  if converted to RGB, default format is BGR
 */
class ClassificationMTImageFeatureToBatch private[bigdl](width: Int, height: Int,
  totalBatchSize: Int, transformer: FeatureTransformer, toRGB: Boolean = false)
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


object RoiImageInfo {
  // the keys in the target table
  // fields from RoiLabel
  val CLASSES = "classes"
  val BBOXES = "bboxes"
  val MASKS = "masks"
  // ISCROWD and ORIGSIZE are stored in ImageFeature
  val ISCROWD = "is_crowd"
  val ORIGSIZE = "orig_size"
  val SCORES = "scores"
  val IMGINFO = "imginfo"

  /**
   * Get the output score tensor from the table.
   *    (1 x N) tensor for N detections
   *
   * @param tab
   * @return
   */
  def getScores(tab: Table): Tensor[Float] = tab[Tensor[Float]](SCORES)

  /**
   * Get the class label tensor from the table. See RoiLabel.classes
   *    the categories for each detections (see RoiLabel.clasees field)
   *    (1 x N), or (2 x N) Tensor[Float]
   *
   * @param tab
   * @return
   */
  def getClasses(tab: Table): Tensor[Float] = tab[Tensor[Float]](CLASSES)

  /**
   * Get the bbox tensor from the table. See RoiLabel.bboxes
   * @param tab
   * @return
   */
  def getBBoxes(tab: Table): Tensor[Float] = tab[Tensor[Float]](BBOXES)

  /**
   * Get the (optional) mask data from the table. See RoiLabel.masks
   * @param tab
   * @return
   */
  def getMasks(tab: Table): Array[RLEMasks] = tab[Array[RLEMasks]](MASKS)

  /**
   * Get the isCrowd tensor from the table. Should be 1 x N vector (N is the # of detections)
   * @param tab
   * @return
   */
  def getIsCrowd(tab: Table): Tensor[Float] = tab[Tensor[Float]](ISCROWD)

  /**
   * Get the size of the image before resizing
   * @return (height, width, channel)
   */
  def getOrigSize(tab: Table): (Int, Int, Int) = tab[(Int, Int, Int)](ORIGSIZE)

  /**
   * Get the isCrowd tensor from the table. Should be 1 x N vector (N is the # of detections)
   * @param tab
   * @return
   */
  def getImageInfo(tab: Table): Tensor[Float] = tab[Tensor[Float]](IMGINFO)

}
/**
 * A batch of images with flattened RoiLabels
 * the getTarget() returns a Table with key from 1 to batchSize. Each key in the table is mapped to
 * a Table for the annotation of an image in the batch. The annotation table holds the annotation
 * info for one image (assume the image has N detections). The annotation table has
 *
 * Key                Value
 * RoiImageInfo.CLASSES   the categories for each detections (see RoiLabel.clasees field)
 *                    (1 x N), or (2 x N) Tensor[Float]
 * RoiImageInfo.BBOXES    the bboxes, (N x 4) Tensor[Float]
 * RoiImageInfo.MASKS     (Optional) the mask data, Array[Tensor[Float]\]. The outer array has N
 *                    elements. The inner tensor holds the data for segmentation
 * RoiImageInfo.ISCROWD   Whether each detection is crowd. (1 x N) Tensor[Float].
 *                    -1: unknown, 0: not crowd, 1: is crowd
 * RoiImageInfo.IMGINFO  with shape (batchSize, 4), contains all images info
 *                 (height, width, original height, original width)
 */
class RoiMiniBatch(val input: Tensor[Float], val target: Array[RoiLabel],
  val isCrowd: Array[Tensor[Float]], val imageInfo: Tensor[Float] = null)
  extends MiniBatch[Float] {

  override def size(): Int = input.size(1)

  override def getInput(): Activity = {
    if (imageInfo == null) input else T(input, imageInfo)
  }

  override def getTarget(): Table = {
    require(target != null, "The target should not be null")
    val tables = (target, isCrowd, 1 to isCrowd.length).zipped.map { case (roiLabel, crowd, i) =>
      val ret = roiLabel.toTable
        .update(RoiImageInfo.ISCROWD, crowd)
      if (imageInfo != null) {
        ret.update(RoiImageInfo.IMGINFO, imageInfo.select(1, i))
      }
      ret
    }
    T.seq(tables)
  }

  override def slice(offset: Int, length: Int): MiniBatch[Float] = {
    val subInput = input.narrow(1, offset, length)
    val subTarget = if (target != null) {
      target.slice(offset - 1, offset + length - 1) // offset starts from 1
    } else {
      null
    }
    val subIsCrowd = isCrowd.slice(offset - 1, offset + length - 1) // offset starts from 1
    val subSize = if (imageInfo != null) imageInfo.narrow(1, offset, length) else null
    RoiMiniBatch(subInput, subTarget, subIsCrowd, subSize)
  }

  override def set(samples: Seq[Sample[Float]])(implicit ev: TensorNumeric[Float])
  : RoiMiniBatch.this.type = {
    throw new NotImplementedError("do not use Sample here")
  }
}

object RoiMiniBatch {
  def apply(data: Tensor[Float], target: Array[RoiLabel],
    isCrowd: Array[Tensor[Float]], imageInfo: Tensor[Float] = null):
  RoiMiniBatch = new RoiMiniBatch(data, target, isCrowd, imageInfo)
}


/**
 * A transformer pipeline wrapper to create RoiMiniBatch in multiple threads
 * The output "target" is a Table. The keys are from 1 to sizeof(batch). The values are
 * the tables for each RoiLabel. Each Roi label table, contains fields of RoiLabel class.
 * The sizes of the input images should be the same
 * @param width final image width
 * @param height final image height
 * @param totalBatchSize global batch size
 * @param transformer pipeline for pre-processing
 * @param toRGB  if converted to RGB, default format is BGR
 */
class RoiImageFeatureToBatch private[bigdl](width: Int, height: Int,
  totalBatchSize: Int, transformer: FeatureTransformer, toRGB: Boolean = false)
  extends MTImageFeatureToBatch(totalBatchSize, transformer) {

  private val frameLength = height * width
  private val featureData: Array[Float] = new Array[Float](batchSize * frameLength * 3)
  private val labelData: Array[RoiLabel] = new Array[RoiLabel](batchSize)
  private val isCrowdData: Array[Tensor[Float]] = new Array[Tensor[Float]](batchSize)
  private val imgInfoData: Tensor[Float] = Tensor[Float](batchSize, 4)
  private var featureTensor: Tensor[Float] = Tensor[Float]()

  override protected def processImageFeature(img: ImageFeature, position: Int): Unit = {
    img.copyTo(featureData, position * frameLength * 3, toRGB = toRGB)
    val isCrowd = img(RoiImageInfo.ISCROWD).asInstanceOf[Tensor[Float]]
    val label = img.getLabel.asInstanceOf[RoiLabel]
    if (label != null) {
      require(isCrowd != null && label.bboxes.size(1) == isCrowd.size(1), "The number" +
        " of detections " +
        "in ImageFeature's ISCROWD should be equal to the number of detections in the RoiLabel")
    } else {
      require(isCrowd == null, "ImageFeature's ISCROWD should be not be set if the label is empty")
    }
    isCrowdData(position) = isCrowd
    labelData(position) = label
    imgInfoData.setValue(position + 1, 1, img.getHeight())
    imgInfoData.setValue(position + 1, 2, img.getWidth())
    imgInfoData.setValue(position + 1, 3, img.getOriginalHeight)
    imgInfoData.setValue(position + 1, 4, img.getOriginalWidth)
  }

  override protected def createBatch(curBatchSize: Int): MiniBatch[Float] = {
    if (featureTensor.nElement() != curBatchSize) {
      featureTensor.set(Storage[Float](featureData),
        storageOffset = 1, sizes = Array(curBatchSize, 3, height, width))
    }

    val labels = checkLabels(arraySlice(labelData, curBatchSize))
    val crowd = if (labels != null) arraySlice(isCrowdData, curBatchSize) else null
    RoiMiniBatch(featureTensor, labels, crowd,
      imgInfoData.narrow(1, 1, curBatchSize))
  }
}

/**
 * A transformer pipeline wrapper to create RoiMiniBatch in multiple threads.
 * Image features may have different sizes, so firstly we need to calculate max size in one batch,
 * then padding all features to one batch with max size.
 * @param sizeDivisible when it's greater than 0,
 *                      height and wide will be round up to multiple of this divisible size
 * @param totalBatchSize global batch size
 * @param transformer pipeline for pre-processing
 * @param toRGB
 */
class RoiImageFeatureToBatchWithResize private[bigdl](sizeDivisible: Int = -1, totalBatchSize: Int,
  transformer: FeatureTransformer, toRGB: Boolean = false)
  extends MTImageFeatureToBatch(totalBatchSize, transformer) {

  private val labelData: Array[RoiLabel] = new Array[RoiLabel](batchSize)
  private val isCrowdData: Array[Tensor[Float]] = new Array[Tensor[Float]](batchSize)
  private val imgInfoData: Tensor[Float] = Tensor[Float](batchSize, 4)
  private var featureTensor: Tensor[Float] = null
  private val imageBuffer = new Array[Tensor[Float]](batchSize)

  private def getFrameSize(batchSize: Int): (Int, Int) = {
    var maxHeight = 0
    var maxWide = 0
    for (i <- 0 until batchSize) {
      maxHeight = math.max(maxHeight, imageBuffer(i).size(2))
      maxWide = math.max(maxWide, imageBuffer(i).size(3))
    }

    if (sizeDivisible > 0) {
      maxHeight = (math.ceil(maxHeight.toFloat / sizeDivisible) * sizeDivisible).toInt
      maxWide = (math.ceil(maxWide.toFloat / sizeDivisible) * sizeDivisible).toInt
    }
    (maxHeight, maxWide)
  }

  override protected def processImageFeature(img: ImageFeature, position: Int): Unit = {
    if (imageBuffer(position) == null) imageBuffer(position) = Tensor[Float]()
    imageBuffer(position).resize(3, img.getHeight(), img.getWidth())
    // save img to buffer
    img.copyTo(imageBuffer(position).storage().array(), 0, toRGB = toRGB)
    val isCrowd = img(RoiImageInfo.ISCROWD).asInstanceOf[Tensor[Float]]
    val label = img.getLabel.asInstanceOf[RoiLabel]
    if (label != null) {
      require(isCrowd != null && label.bboxes.size(1) == isCrowd.size(1), "The number of " +
        "detections in ImageFeature's ISCROWD should be equal to the number of detections in the " +
        "RoiLabel")
    } else {
      require(isCrowd == null, "ImageFeature's ISCROWD should be not be set if the label is empty")
    }
    isCrowdData(position) = isCrowd
    labelData(position) = label
    imgInfoData.setValue(position + 1, 1, img.getHeight())
    imgInfoData.setValue(position + 1, 2, img.getWidth())
    imgInfoData.setValue(position + 1, 3, img.getOriginalHeight)
    imgInfoData.setValue(position + 1, 4, img.getOriginalWidth)
  }

  override protected def createBatch(batchSize: Int): MiniBatch[Float] = {
    val (height, wide) = getFrameSize(batchSize)
    if (featureTensor == null) featureTensor = Tensor()
    featureTensor.resize(batchSize, 3, height, wide).fill(0.0f)
    // copy img buffer to feature tensor
    for (i <- 0 until batchSize) {
      featureTensor.select(1, i + 1).narrow(2, 1, imageBuffer(i).size(2))
        .narrow(3, 1, imageBuffer(i).size(3)).copy(imageBuffer(i))
    }

    val labels = checkLabels(arraySlice(labelData, batchSize))
    val crowd = if (labels != null) arraySlice(isCrowdData, batchSize) else null
    RoiMiniBatch(featureTensor, labels, crowd, imgInfoData.narrow(1, 1, batchSize))
  }
}
