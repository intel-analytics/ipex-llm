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

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.segmentation.{MaskUtils, RLEMasks}
import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.nn.AbsCriterion
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.RoiImageInfo
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import com.intel.analytics.bigdl.utils.Table
import org.apache.commons.lang3.SerializationUtils
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * A method defined to evaluate the model.
 * This trait can be extended by user-defined method. Such
 * as Top1Accuracy
 */
trait ValidationMethod[T] extends Serializable {
  def apply(output: Activity, target: Activity): ValidationResult

  // return the name of this method
  protected def format(): String

  // return the name of this method
  override def toString(): String = format()

  // deep clone the object
  override def clone(): ValidationMethod[T] = SerializationUtils.clone(this)
}

/**
 * A result that calculate the numeric value of a validation method.
 * User-defined valuation results must override the + operation and result() method.
 * It is executed over the samples in each batch.
 */
trait ValidationResult extends Serializable {

  // return the calculation results over all the samples in the batch
  def result(): (Float, Int) // (Result, TotalNum)

  // scalastyle:off methodName
  def +(other: ValidationResult): ValidationResult

  // return the name of this trait
  protected def format(): String

  // return the name of this trait
  override def toString(): String = format()
}

/**
 * Represent an accuracy result. Accuracy means a ratio of correct number and total number.
 * @param correct correct number
 * @param count total count number
 */
class AccuracyResult(private var correct: Int, private var count: Int)
  extends ValidationResult {

  override def result(): (Float, Int) = (correct.toFloat/count, count)

  // scalastyle:off methodName
  override def +(other: ValidationResult): ValidationResult = {
    val otherResult = other.asInstanceOf[AccuracyResult]
    this.correct += otherResult.correct
    this.count += otherResult.count
    this
  }
  // scalastyle:on methodName


  override protected def format(): String = {
    s"Accuracy(correct: $correct, count: $count, accuracy: ${correct.toDouble / count})"
  }

  override def equals(obj: Any): Boolean = {
    if (obj == null) {
      return false
    }
    if (!obj.isInstanceOf[AccuracyResult]) {
      return false
    }
    val other = obj.asInstanceOf[AccuracyResult]
    if (this.eq(other)) {
      return true
    }
    this.correct == other.correct && this.count == other.count
  }

  override def hashCode(): Int = {
    val seed = 37
    var hash = 1
    hash = hash * seed + this.correct
    hash = hash * seed + this.count
    hash
  }
}

/**
 * This is a metric to measure the accuracy of Tree Neural Network/Recursive Neural Network
 *
 */
class TreeNNAccuracy[T: ClassTag]()(
  implicit ev: TensorNumeric[T])
  extends ValidationMethod[T] {
  override def apply(output: Activity, target: Activity):
  ValidationResult = {
    var correct = 0
    var count = 0

    var _output = output.asInstanceOf[Tensor[T]]
    val _target = target.asInstanceOf[Tensor[T]].select(2, 1)

    if (_output.dim() == 3) {
      _output = _output.select(2, 1)
      (if (_output.size(2) == 1) {
        _output.clone().apply1(x => if (ev.isGreater(ev.fromType(0.5), x)) ev.zero else ev.one)
      } else {
        _output.max(2)._2.squeeze()
      }).map(_target, (a, b) => {
        if (a == b) {
          correct += 1
        }
        a
      })
      count += _output.size(1)
    } else if (_output.dim == 2) {
      _output = _output.select(1, 1)
      require(_target.size(1) == 1)
      (if (_output.size(1) == 1) {
        _output.clone().apply1(x => if (ev.isGreater(ev.fromType(0.5), x)) ev.zero else ev.one)
      } else {
        _output.max(1)._2.squeeze()
      }).map(_target, (a, b) => {
        if (a == b) {
          correct += 1
        }
        a
      })
      count += 1
    } else {
      throw new IllegalArgumentException
    }

    new AccuracyResult(correct, count)
  }

  override def format(): String =
    s"TreeNNAccuracy()"
}

/**
 * Caculate the percentage that output's max probability index equals target
 */
class Top1Accuracy[T: ClassTag](
  implicit ev: TensorNumeric[T])
  extends ValidationMethod[T] {
  override def apply(output: Activity, target: Activity):
  ValidationResult = {
    var correct = 0
    var count = 0

    val _target = target.asInstanceOf[Tensor[T]]
    val _output = if (output.toTensor[T].nDimension() != 1 &&
      output.toTensor[T].size().head != _target.size().head) {
      output.toTensor[T].narrow(1, 1, _target.size().head)
    } else {
      output.toTensor[T]
    }

    if (_output.dim() == 2) {
      (if (_output.size(2) == 1) {
        _output.clone().apply1(x => if (ev.isGreater(ev.fromType(0.5), x)) ev.zero else ev.one)
      } else {
        _output.max(2)._2.squeeze()
      }).map(_target, (a, b) => {
        if (a == b) {
          correct += 1
        }
        a
      })
      count += _output.size(1)
    } else if (_output.dim == 1) {
      require(_target.size(1) == 1)
      (if (_output.size(1) == 1) {
        _output.clone().apply1(x => if (ev.isGreater(ev.fromType(0.5), x)) ev.zero else ev.one)
      } else {
        _output.max(1)._2
      }).map(_target, (a, b) => {
        if (a == b) {
          correct += 1
        }
        a
      })
      count += 1
    } else {
      throw new IllegalArgumentException
    }

    new AccuracyResult(correct, count)
  }

  override def format(): String = "Top1Accuracy"
}

/**
 * Calculate the Mean Average Precision (MAP). The algorithm follows VOC Challenge after 2007
 * Require class label beginning with 0
 * @param k Take top-k confident predictions into account. If k=-1, calculate on all predictions
 * @param classes The number of classes
 */
class MeanAveragePrecision[T: ClassTag](k: Int, classes: Int)(
  implicit ev: TensorNumeric[T]) extends ValidationMethod[T] {

  require(classes > 0 && classes <= classes, s"The number of classes should be "
    + s"> 0 and <= $classes, but got $classes")
  require(k > 0, s"k should be > 0, but got $k")

  override def apply(output: Activity, target: Activity): ValidationResult = {
    var _target = target.asInstanceOf[Tensor[T]].squeezeNewTensor()

    val outTensor = output.toTensor[T]
    val _output = if (outTensor.nDimension() != 1 &&
      outTensor.size(1) != _target.size(1)) {
      outTensor.narrow(1, 1, _target.size().head)
    } else {
      outTensor
    }

    require(_output.dim()==1 && _target.nElement() == 1 ||
      _output.size(1) == _target.nElement(), "The number of samples in the output should " +
      "be the same as in the target")

    val posCnt = new Array[Int](classes)
    for (i <- 1 to _target.nElement()) {
      val clazz = ev.toType[Float](_target.valueAt(i))
      require(clazz == math.ceil(clazz), s"The class for $i-th test sample should be an integer, "
        + s"got $clazz")
      val intClazz = clazz.toInt
      require(intClazz >= 0 && intClazz < classes, s"The class for $i-th test sample should be "
        + s">= 0 and < $classes, but got $intClazz")
      posCnt(intClazz) += 1
    }

    val confidenceArr = (0 until classes).map(_ => new ArrayBuffer[(Float, Boolean)]).toArray
    if (_output.nDimension() == 2) {
      (1 to _output.size(1)).foreach(i => {
        val row = _output.select(1, i)
        val gtClz = ev.toType[Float](_target.valueAt(i))
        for(clz <- 0 until classes) {
          confidenceArr(clz) += ((ev.toType[Float](row.valueAt(clz + 1)), gtClz == clz))
        }
      })
    } else {
      require(_output.dim() == 1, "The output should have 1 or 2 dimensions")
      val row = _output
      val gtClz = ev.toType[Float](_target.valueAt(1))
      for(clz <- 0 until classes) {
        confidenceArr(clz) += ((ev.toType[Float](row.valueAt(clz + 1)), gtClz == clz))
      }
    }
    new MAPValidationResult(classes, k, confidenceArr, posCnt)
  }

  override def format(): String = s"MAP@$k"
}

object MAPUtil {

  // find top k values & indices in a column of a matrix
  def findTopK(k: Int, arr: Array[Array[Float]], column: Int): Array[(Int, Float)] = {
    val q = collection.mutable.PriorityQueue[(Int, Float)]()(Ordering.by[(Int, Float), Float](_._2))
    arr.indices.foreach(i => {
      q.enqueue((i, arr(i)(column)))
    })
    val end = Math.min(k, q.size)
    (1 to end).map(_ => q.dequeue()).toArray
  }

  /**
   * convert the ground truth into parsed GroundTruthRegions
   * @param gtTable
   * @param classes
   * @param isCOCO if using COCO's algorithm for IOU computation
   * @param isSegmentation
   * @return (array of GT BBoxes of images, # of GT bboxes for each class)
   */
  def gtTablesToGroundTruthRegions(gtTable: Table, classes: Int, numIOU: Int, isCOCO: Boolean,
    isSegmentation: Boolean): (Array[ArrayBuffer[GroundTruthRegion]], Array[Int]) = {
    // the number of GT bboxes for each class
    val gtCntByClass = new Array[Int](classes)

    // one image may contain multiple Ground truth bboxes
    val gtImages = (1 to gtTable.length()).map { i =>
      val gtImage = new ArrayBuffer[GroundTruthRegion]()
      val roiLabel = gtTable[Table](i)
      if (roiLabel.length() > 0) {
        val bbox = RoiImageInfo.getBBoxes(roiLabel)
        val tclasses = RoiImageInfo.getClasses(roiLabel)
        val isCrowd = RoiImageInfo.getIsCrowd(roiLabel)
        val masks = if (isSegmentation) RoiImageInfo.getMasks(roiLabel) else null
        val bboxCnt = bbox.size(1)
        require(bboxCnt == tclasses.size(1), "CLASSES of target tables should have the" +
          "same size of the bbox counts")
        require(bboxCnt == isCrowd.nElement(), "ISCROWD of target tables should have the" +
          "same size of the bbox counts")
        require(masks == null || bboxCnt == masks.length, "MASKS of target tables should have the" +
          "same size of the bbox counts")
        for (j <- 1 to bboxCnt) {
          val (label, _diff) = if (tclasses.dim() == 2) {
            (tclasses.valueAt(1, j).toInt, tclasses.valueAt(2, j))
          } else {
            (tclasses.valueAt(j).toInt, 0f)
          }
          val diff = if (isCrowd.valueAt(j) != 0 || _diff != 0) 1f else 0f
          val newGt = if (isSegmentation) {
            new GroundTruthRLE(numIOU, label, diff, masks(j - 1))
          } else {
            new GroundTruthBBox(isCOCO, numIOU, label, diff, bbox.valueAt(j, 1),
              bbox.valueAt(j, 2), bbox.valueAt(j, 3), bbox.valueAt(j, 4))
          }
          gtImage += newGt
          require(label >= 0 && label < classes, s"Bad label id $label")

          if (diff == 0) {
            gtCntByClass(label) += 1
          }
        }
      }
      gtImage
    }.toArray
    (gtImages, gtCntByClass)
  }

  /**
   * For a detection, match it with all GT boxes. Record the match in "predictByClass"
   */
  def parseDetection(gtBbox: ArrayBuffer[GroundTruthRegion], label: Int, score: Float, x1: Float,
    y1: Float, x2: Float, y2: Float, mask: RLEMasks, classes: Int, iou: Array[Float],
    predictByClasses: Array[Array[ArrayBuffer[(Float, Boolean)]]]): Unit = {
    require(label >= 0 && label < classes, s"Bad label id $label")
    for (i <- iou.indices) {
      // for each GT boxes, try to find a matched one with current prediction
      val matchedGt = gtBbox.toIterator.filter(gt => label == gt.label && gt.canOccupy(i))
        .flatMap(gt => { // calculate and filter out the bbox
          val iouRate = gt.getIOURate(x1, y1, x2, y2, mask)
          if (iouRate >= iou(i)) Iterator.single((gt, iouRate)) else Iterator.empty
        })
        .reduceOption((gtArea1, gtArea2) => { // find max IOU bbox
          if (gtArea1._1.diff != gtArea2._1.diff) {
            if (gtArea1._1.diff > gtArea2._1.diff) gtArea2 else gtArea1
          } else {
            if (gtArea1._2 > gtArea2._2) gtArea1 else gtArea2
          }
        })
        .map(bbox => { // occupy the bbox
          bbox._1.occupy(i)
          bbox._1
        })
      if (matchedGt.isEmpty || matchedGt.get.diff == 0) {
        predictByClasses(i)(label).append((score, matchedGt.isDefined))
      }
      // else: when the prediction matches a "difficult" GT, do nothing
      // it is neither TP nor FP
      // "difficult" is defined in PASCAL VOC dataset, meaning the image is difficult to detect
    }
  }

  def parseSegmentationTensorResult(outTensor: Tensor[Float],
    func: (Int, Int, Float, Float, Float, Float, Float) => Unit): Unit = {
    require(outTensor.dim() == 2, "the output tensor should have 2 dimensions")
    for (imgId <- 0 until outTensor.size(1)) {
      // for each image
      val batch = outTensor.select(1, imgId + 1)
      val batchSize = batch.valueAt(1).toInt
      var offset = 2
      for (bboxIdx <- 0 until batchSize) {
        // for each predicted bboxes
        val label = batch.valueAt(offset).toInt
        val score = batch.valueAt(offset + 1)
        val x1 = batch.valueAt(offset + 2)
        val y1 = batch.valueAt(offset + 3)
        val x2 = batch.valueAt(offset + 4)
        val y2 = batch.valueAt(offset + 5)
        func(imgId, label, score, x1, y1, x2, y2)
        offset += 6
      }
    }
  }
}

class MAPType extends Serializable
object MAPPascalVoc2007 extends MAPType
object MAPPascalVoc2010 extends MAPType
object MAPCOCO extends MAPType

/**
 * The MAP Validation Result. The results are not calculated until result() or format() is called
 * require class label beginning with 0
 */
class MAPValidationResult(
  private val nClass: Int,
  // take the first k samples, or -1 for all samples
  private val k: Int,
  // the predicts for each classes. (Confidence, GT)
  private[bigdl] var predictForClass: Array[ArrayBuffer[(Float, Boolean)]],
  private[bigdl] var gtCntForClass: Array[Int],
  private val theType: MAPType = MAPPascalVoc2010,
  private val skipClass: Int = -1,
  private val isSegmentation: Boolean = false
)
  extends ValidationResult {

  if (skipClass < 0) {
    require(skipClass == -1, s"Invalid skipClass $skipClass")
  } else {
    require(skipClass >= 0 && skipClass < nClass, s"Invalid skipClass $skipClass")
  }

  private def sortPredictions(p: ArrayBuffer[(Float, Boolean)]): ArrayBuffer[(Float, Boolean)] = {
    p.sortBy(v => v._1)(Ordering.Float.reverse) // decending order
  }

  private[bigdl] def calculateClassAP(clz: Int): Float = {
    val posCnt = gtCntForClass
    // for each class, first find top k confident samples
    val sorted = sortPredictions(predictForClass(clz))
    var tp = 0
    val refinedK = if (k > 0) k else sorted.size
    // calculate the max precision for each different recall
    // for each top-j items, calculate the (precision, recall)
    val PnR = sorted.take(refinedK).zipWithIndex.flatMap { case (predict, j) =>
      if (predict._2) {
        // if it is a hit
        tp += 1
        // j + 1 is the total number of samples marked positive by the model
        val precision = tp.toFloat / (j + 1)
        val recall = tp.toFloat / posCnt(clz)
        Iterator.single(recall, precision)
      } else {
        Iterator.empty
      }
    }

    // get Average precision over each different recall
    theType match {
      case _: MAPPascalVoc2007.type =>
        (0 to 10).map(r => {
          val recall = 0.1f * r
          // for every (R,P), where R>=recall, get max(P)
          PnR.filter(_._1 >= recall).map(_._2).reduceOption(_ max _).getOrElse(0f)
        })
          .reduceOption(_ + _)
          .map(_ / 11)
          .getOrElse(0f)
      case _: MAPPascalVoc2010.type =>
        (1 to posCnt(clz)).map(r => {
          val recall = r.toFloat / posCnt(clz)
          // for every (R,P), where R>=recall, get max(P)
          PnR.filter(_._1 >= recall).map(_._2).reduceOption(_ max _).getOrElse(0f)
        })
          .reduceOption(_ + _)
          .map(_ / posCnt(clz))
          .getOrElse(0f)
      case _: MAPCOCO.type =>
        if (posCnt(clz) == 0) {
          -1f
        } else {
          (0 to 100).map(r => {
            val recall = 0.01f * r
            // for every (R,P), where R>=recall, get max(P)
            PnR.filter(_._1 >= recall).map(_._2).reduceOption(_ max _).getOrElse(0f)
          })
            .reduceOption(_ + _)
            .map(_ / 101)
            .getOrElse(0f)
        }
    }
  }


  override def result(): (Float, Int) = {
    // get the indices of top-k confident samples
    val AP = (0 until nClass).filter(_ != skipClass).map { clz => calculateClassAP(clz) }
    // APs are got. Now we get MAP
    val result = theType match {
      case t: MAPCOCO.type =>
        val filtered = AP.filter(_ != -1f)
        filtered.sum / filtered.length
      case _ => AP.sum / (nClass - (if (skipClass == -1) 0 else 1))
    }
    (result, 1)
  }

  private[optim] def mergeWithoutGtCnt(o: MAPValidationResult): MAPValidationResult = {
    require(predictForClass.length == o.predictForClass.length)
    require(gtCntForClass.length == o.gtCntForClass.length)
    for (i <- predictForClass.indices) {
      val (left, right) = (predictForClass(i), o.predictForClass(i))
      left ++= right
      predictForClass(i) = if (k < 0) {
        left
      } else {
        val sorted = sortPredictions(left)
        sorted.take(k)
      }
    }
    this
  }

  // scalastyle:off methodName
  override def +(other: ValidationResult): ValidationResult = {
    val o = other.asInstanceOf[MAPValidationResult]
    mergeWithoutGtCnt(o)
    gtCntForClass.indices.foreach( i => gtCntForClass(i) += o.gtCntForClass(i))
    this
  }
  // scalastyle:on methodName

  override protected def format(): String = {
    val segOrBbox = if (isSegmentation) "segm" else "bbox"
    val resultStr = (0 until nClass).map { clz => calculateClassAP(clz) }.zipWithIndex
      .map { t => s"AP of class ${t._2} = ${t._1}\n"}.reduceOption( _ + _).getOrElse("")
    s"MeanAveragePrecision_$segOrBbox@$k(${result()._1})\n $resultStr"
  }
}

abstract private[bigdl] class GroundTruthRegion(isCOCO: Boolean, numIOU: Int, val label: Int,
  val diff: Float) {
  // if is false, the region is not matched with any predictions
  // indexed by the IOU threshold index
  private val isOccupied = new Array[Boolean](numIOU)

  /**
   * Returns if any previous prediction is matched with the current region
   *
   * @return
   */
  def canOccupy(iouIdx: Int): Boolean = (isCOCO && diff == 1) || !isOccupied(iouIdx)

  def occupy(iouIdx: Int): Unit = {
    isOccupied(iouIdx) = true
  }

  /** get the IOU rate of another region with the current region
   *
   * @param x1 the min x
   * @param y1 the min y
   * @param x2 the max x
   * @param y2 the max y
   * @param rle RLE mask data, can be null
   * @return
   */
  def getIOURate(x1: Float, y1: Float, x2: Float, y2: Float, rle: RLEMasks = null): Float
}

private[bigdl] class GroundTruthBBox(isCOCO: Boolean, numIOU: Int, label: Int, diff: Float,
  val xmin: Float, val ymin: Float, val xmax: Float, val ymax: Float)
  extends GroundTruthRegion(isCOCO, numIOU, label, diff) {
  private val area = (xmax - xmin + 1) * (ymax - ymin + 1)

  override def getIOURate(x1: Float, y1: Float, x2: Float, y2: Float,
      rle: RLEMasks = null): Float = {
    val ixmin = Math.max(xmin, x1)
    val iymin = Math.max(ymin, y1)
    val ixmax = Math.min(xmax, x2)
    val iymax = Math.min(ymax, y2)
    val inter = Math.max(ixmax - ixmin + 1, 0) * Math.max(iymax - iymin + 1, 0)
    val detectionArea = (x2 - x1 + 1) * (y2 - y1 + 1)
    val union = if (isCOCO && diff != 0) detectionArea else (detectionArea + area - inter)
    inter / union
  }
}

private[bigdl] class GroundTruthRLE(numIOU: Int, label: Int, diff: Float, rle: RLEMasks)
  extends GroundTruthRegion(true, numIOU, label, diff) {

  override def getIOURate(x1: Float, y1: Float, x2: Float, y2: Float,
    detRLE: RLEMasks): Float = {
    MaskUtils.rleIOU(detRLE, rle, diff != 0)
  }
}

class MAPMultiIOUValidationResult(
  private val nClass: Int,
  // take the first k samples, or -1 for all samples
  private val k: Int,
  // the predicts for each classes.
  // predictForClassIOU(iouIdx)(cls) is an array of (Confidence, GT)
  private val predictForClassIOU: Array[Array[ArrayBuffer[(Float, Boolean)]]],
  private var gtCntForClass: Array[Int],
  private val iouRange: (Float, Float),
  private val theType: MAPType = MAPPascalVoc2010,
  private val skipClass: Int = -1,
  private val isSegmentation: Boolean = false) extends ValidationResult {

  val impl = predictForClassIOU.map(predictForClass => {
    new MAPValidationResult(nClass, k, predictForClass,
      gtCntForClass, theType, skipClass, isSegmentation)
  })
  override def result(): (Float, Int) = (impl.map(_.result()._1).sum / impl.length, 1)

  // scalastyle:off methodName
  override def +(other: ValidationResult): ValidationResult = {
    val o = other.asInstanceOf[MAPMultiIOUValidationResult]
    require(o.predictForClassIOU.length == predictForClassIOU.length,
      "To merge MAPMultiIOUValidationResult, the length of predictForClassIOU should be" +
        "the same")
    impl.zip(o.impl).foreach { case (v1, v2) => v1.mergeWithoutGtCnt(v2) }
    gtCntForClass.indices.foreach( i => gtCntForClass(i) += o.gtCntForClass(i))
    this
  }
  // scalastyle:on methodName

  override protected def format(): String = {
    val step = (iouRange._2 - iouRange._1) / (predictForClassIOU.length - 1)
    val results = impl.map(_.result()._1)
    val resultStr = results.zipWithIndex
      .map { t => s"\t IOU(${iouRange._1 + t._2 * step}) = ${t._1}\n"}
      .reduceOption( _ + _).getOrElse("")
    val segOrBbox = if (isSegmentation) "segm" else "bbox"
    f"MAP_$segOrBbox@IOU(${iouRange._1}%1.3f:$step%1.3f:${iouRange._2}%1.3f)=" +
      s"${results.sum / impl.length}\n$resultStr"
  }
}

/** MeanAveragePrecision for Object Detection
 * The class label begins with 0
 *
 * The expected output from the last layer should be a Tensor[Float] or a Table
 * If output is a tensor, it should be [num_of_batch X (1 + maxDetection * 6)] matrix
 * The format of the matrix should be [<batch>, <batch>, ...], where each row vector is
 * <batch> = [<size_of_batch>, <sample>,...]. Each sample has format:
 * <sample> = <label, score, bbox x4>
 * imgId is the batch number of the sample. imgId begins with 0.
 * Multiple samples may share one imgId
 *
 * If output is a table, it is a table of tables.
 * output(i) is the results of the i-th image in the batch, where i = 1 to sizeof(batch)
 * output(i) is a table, which contains the same keys (fields) of image info in the "target"
 * Please refer to RoiMiniBatch/RoiImageInfo's documents. Besides, the inner tables also contain
 * the scores for the detections in the image.
 *
 * The "target" (Ground truth) is a table with the same structure of "output", except that
 * it does not have "score" field
 *
 * @param classes the number of classes
 * @param topK only take topK confident predictions (-1 for all predictions)
 * @param iouThres the IOU thresholds
 * @param theType the type of MAP algorithm. (voc2007/voc2010/COCO)
 * @param skipClass skip calculating on a specific class (e.g. background)
 *                  the class index starts from 0, or is -1 if no skipping
 * @param isSegmentation if check the IOU of segmentations instead of bounding boxes. If true,
 *                       the output and target must have "masks" data
 */
class MeanAveragePrecisionObjectDetection[T: ClassTag](
  classes: Int, topK: Int = -1, iouThres: Array[Float] = Array(0.5f),
  theType: MAPType = MAPPascalVoc2010, skipClass: Int = -1, isSegmentation: Boolean = false)(
  implicit ev: TensorNumeric[T]) extends ValidationMethod[T] {
  override def apply(output: Activity, target: Activity): ValidationResult = {
    // one image may contain multiple Ground truth bboxes
    val (gtImages, gtCntByClass) =
      MAPUtil.gtTablesToGroundTruthRegions(target.toTable, classes, iouThres.length,
        theType.isInstanceOf[MAPCOCO.type], isSegmentation)

    // the predicted bboxes for each classes
    // predictByClasses(iouIdx)(classIdx)(bboxNum) is (Confidence, GT)
    val predictByClasses = iouThres.map(_iou => {
      (0 until classes).map(_ => new ArrayBuffer[(Float, Boolean)]).toArray
    })

    output match {
      case _outTensor: Tensor[_] =>
        require(!isSegmentation, "Cannot get segmentation data from tensor output for MAP")
        val outTensor = _outTensor.asInstanceOf[Tensor[Float]]
        MAPUtil.parseSegmentationTensorResult(outTensor,
          (imgIdx, label, score, x1, y1, x2, y2) => {
            val gtBbox = gtImages(imgIdx)
            MAPUtil.parseDetection(gtBbox, label, score, x1, y1, x2, y2, null, classes, iouThres,
              predictByClasses = predictByClasses)
          })
      case outTable: Table =>
        require(gtImages.length == outTable.length(), "The number of images in the output and " +
          "in the target should be the same")
        for (imgId <- 1 to outTable.length()) {
          val gtBbox = gtImages(imgId - 1)
          val imgOut = outTable[Table](imgId)
          // if the image contains empty predictions, do nothing
          if (imgOut.length() > 0) {
            val bboxes = RoiImageInfo.getBBoxes(imgOut)
            val scores = RoiImageInfo.getScores(imgOut)
            val labels = RoiImageInfo.getClasses(imgOut)
            require(bboxes.dim() == 2, "the bbox tensor should have 2 dimensions")
            val masks = if (isSegmentation) Some(RoiImageInfo.getMasks(imgOut)) else None
            val batchSize = bboxes.size(1)
            require(batchSize == labels.size(1), "CLASSES of target tables should have the" +
              "same size of the bbox counts")
            require(batchSize == scores.nElement(), "ISCROWD of target tables should have the" +
              "same size of the bbox counts")
            require(masks.isEmpty || batchSize == masks.get.length, "MASKS of target tables " +
              "should have the same size of the bbox counts")
            val detections = new ArrayBuffer[(Int, Float, Float, Float, Float,
              Float, RLEMasks)]()
            for (bboxIdx <- 1 to batchSize) {
              val score = scores.valueAt(bboxIdx)
              val x1 = bboxes.valueAt(bboxIdx, 1)
              val y1 = bboxes.valueAt(bboxIdx, 2)
              val x2 = bboxes.valueAt(bboxIdx, 3)
              val y2 = bboxes.valueAt(bboxIdx, 4)
              val label = labels.valueAt(bboxIdx).toInt
              val mask = masks.map(_ (bboxIdx - 1)).orNull
              detections.append((label, score, x1, y1, x2, y2, mask))
            }
            detections.sortBy(v => v._2)(Ordering.Float.reverse).foreach {
              case (label, score, x1, y1, x2, y2, mask) =>
                MAPUtil.parseDetection(gtBbox, label, score, x1, y1, x2, y2, mask, classes,
                  iouThres, predictByClasses)
            }
          }
        }
    }
    if (iouThres.length != 1) {
      new MAPMultiIOUValidationResult(classes, topK, predictByClasses, gtCntByClass,
        (iouThres.head, iouThres.last), theType, skipClass, isSegmentation)
    } else {
      new MAPValidationResult(classes, topK, predictByClasses.head, gtCntByClass, theType,
        skipClass, isSegmentation)
    }
  }

  override protected def format(): String = s"MAPObjectDetection"
}

object MeanAveragePrecision {
  /**
   * Create MeanAveragePrecision validation method using COCO's algorithm for object detection.
   * IOU computed by the segmentation masks
   *
   * @param nClasses the number of classes (including skipped class)
   * @param topK only take topK confident predictions (-1 for all predictions)
   * @param skipClass skip calculating on a specific class (e.g. background)
   *                  the class index starts from 0, or is -1 if no skipping
   * @param iouThres the IOU thresholds, (rangeStart, stepSize, numOfThres), inclusive
   * @return MeanAveragePrecisionObjectDetection
   */
  def cocoSegmentation(nClasses: Int, topK: Int = -1, skipClass: Int = 0,
    iouThres: (Float, Float, Int) = (0.5f, 0.05f, 10))
  : MeanAveragePrecisionObjectDetection[Float] = {
    createCOCOMAP(nClasses, topK, skipClass, iouThres, true)
  }

  /**
   * Create MeanAveragePrecision validation method using COCO's algorithm for object detection.
   * IOU computed by the bounding boxes
   *
   * @param nClasses the number of classes (including skipped class)
   * @param topK only take topK confident predictions (-1 for all predictions)
   * @param skipClass skip calculating on a specific class (e.g. background)
   *                  the class index starts from 0, or is -1 if no skipping
   * @param iouThres the IOU thresholds, (rangeStart, stepSize, numOfThres), inclusive
   * @return MeanAveragePrecisionObjectDetection
   */
  def cocoBBox(nClasses: Int, topK: Int = -1, skipClass: Int = 0,
    iouThres: (Float, Float, Int) = (0.5f, 0.05f, 10))
  : MeanAveragePrecisionObjectDetection[Float] = {
    createCOCOMAP(nClasses, topK, skipClass, iouThres, false)
  }

  /**
   * Calculate the Mean Average Precision (MAP) for classification output and target
   * The algorithm follows VOC Challenge after 2007
   * Require class label beginning with 0
   *
   * @param nClasses The number of classes
   * @param topK Take top-k confident predictions into account. If k=-1,calculate on all predictions
   */
  def classification(nClasses: Int, topK: Int = -1)
  : MeanAveragePrecision[Float] = new MeanAveragePrecision[Float](topK, nClasses)

  private def createCOCOMAP(nClasses: Int, topK: Int, skipClass: Int,
    iouThres: (Float, Float, Int), isSegmentation: Boolean)
  : MeanAveragePrecisionObjectDetection[Float] = {
    new MeanAveragePrecisionObjectDetection[Float](nClasses, topK,
      (0 until iouThres._3).map(iouThres._1 + _ * iouThres._2).toArray,
      MAPCOCO, skipClass, isSegmentation)
  }

  /**
   * Create MeanAveragePrecision validation method using Pascal VOC's algorithm for object detection
   *
   * @param nClasses the number of classes
   * @param useVoc2007 if using the algorithm in Voc2007 (11 points). Otherwise, use Voc2010
   * @param topK only take topK confident predictions (-1 for all predictions)
   * @param skipClass skip calculating on a specific class (e.g. background)
   *                  the class index starts from 0, or is -1 if no skipping
   * @return MeanAveragePrecisionObjectDetection
   */
  def pascalVOC(nClasses: Int, useVoc2007: Boolean = false, topK: Int = -1,
    skipClass: Int = 0) : MeanAveragePrecisionObjectDetection[Float] = {
    new MeanAveragePrecisionObjectDetection[Float](nClasses, topK,
      theType = if (useVoc2007) MAPPascalVoc2007 else MAPPascalVoc2010,
      skipClass = skipClass)
  }
}

/**
 * Calculate the percentage that target in output's top5 probability indexes
 */
class Top5Accuracy[T: ClassTag](
  implicit ev: TensorNumeric[T]) extends ValidationMethod[T] {
  override def apply(output: Activity, target: Activity):
  AccuracyResult = {
    var _target = target.asInstanceOf[Tensor[T]].squeezeNewTensor()

    val _output = if (output.toTensor[T].nDimension() != 1 &&
      output.toTensor[T].size(1) != _target.size(1)) {
      output.toTensor[T].narrow(1, 1, _target.size().head)
    } else {
      output.toTensor[T]
    }

    var correct = 0
    var count = 0
    if (_output.dim() == 2) {
      val indices = _output.topk(5, 2, false)._2
      var i = 1
      while (i <= _output.size(1)) {
        if (indices.valueAt(i, 1) == _target.valueAt(i)
          || indices.valueAt(i, 2) == _target.valueAt(i)
          || indices.valueAt(i, 3) == _target.valueAt(i)
          || indices.valueAt(i, 4) == _target.valueAt(i)
          || indices.valueAt(i, 5) == _target.valueAt(i)) {
          correct += 1
        }
        i += 1
      }
      count += _output.size(1)
    } else if (_output.dim == 1) {
      require(_target.size(1) == 1)
      val indices = _output.topk(5, 1, false)._2
      if (indices.valueAt(1) == _target.valueAt(1) || indices.valueAt(2) == _target.valueAt(1)
        || indices.valueAt(3) == _target.valueAt(1) || indices.valueAt(4) == _target.valueAt(1)
        || indices.valueAt(5) == _target.valueAt(1)) {
        correct += 1
      }
      count += 1
    } else {
      throw new IllegalArgumentException
    }

    new AccuracyResult(correct, count)
  }

  override def format(): String = "Top5Accuracy"
}

/**
 * Hit Ratio(HR).
 * HR intuitively measures whether the test item is present on the top-k list.
 *
 * @param k top k.
 * @param negNum number of negative items.
 */
class HitRatio[T: ClassTag](k: Int = 10, negNum: Int = 100)(
    implicit ev: TensorNumeric[T])
  extends ValidationMethod[T] {
  /**
   * Output and target should belong to the same user.
   * And have (negNum + 1) elements.
   * Target should have only one positive label, means one element is 1, others
   * are all 0.
   * A couple of output and target will be count as one record.
   */
  override def apply(output: Activity, target: Activity): ValidationResult = {
    require(output.toTensor[T].nElement() == negNum + 1,
      s"negNum is $negNum, output's nElement should be ${negNum}, but got" +
        s" ${output.toTensor[T].nElement()}")
    require(target.toTensor[T].nElement() == negNum + 1,
      s"negNum is $negNum, target's nElement should be ${negNum}, but got" +
        s" ${output.toTensor[T].nElement()}")
    val o = output.toTensor[T].resize(1 + negNum)
    val t = target.toTensor[T].resize(1 + negNum)
    var positiveItem = 0
    var positiveCount = 0
    var i = 1
    while(i <= t.nElement()) {
      if (t.valueAt(i) == 1) {
        positiveItem = i
        positiveCount += 1
      }
      i += 1
    }
    require(positiveItem != 0, s"${format()}: no positive item.")
    require(positiveCount == 1, s"${format()}: too many positive items, excepted 1," +
      s" but got $positiveCount")

    val hr = calHitRate(positiveItem, o, k)

    new ContiguousResult(hr, 1, s"HitRatio@$k")
  }

  // compute hit rate
  private def calHitRate(index: Int, o: Tensor[T], k: Int): Float = {
    var topK = 1
    var i = 1
    val precision = ev.toType[Float](o.valueAt(index))
    while (i < o.nElement() && topK <= k) {
      if (ev.toType[Float](o.valueAt(i)) > precision) {
        topK += 1
      }
      i += 1
    }

    if(topK <= k) {
      1
    } else {
      0
    }
  }

  override def format(): String = "HitRate@10"
}

/**
 * Normalized Discounted Cumulative Gain(NDCG).
 * NDCG accounts for the position of the hit by assigning higher scores to hits at top ranks.
 *
 * @param k top k.
 * @param negNum number of negative items.
 */
class NDCG[T: ClassTag](k: Int = 10, negNum: Int = 100)(
    implicit ev: TensorNumeric[T])
  extends ValidationMethod[T] {
  /**
   * Output and target should belong to the same user.
   * And have (negNum + 1) elements.
   * Target should have only one positive label, means one element is 1, others
   * are all 0.
   * A couple of output and target will be count as one record.
   */
  override def apply(output: Activity, target: Activity): ValidationResult = {
    require(output.toTensor[T].nElement() == negNum + 1,
      s"negNum is $negNum, output's nElement should be ${negNum}, but got" +
        s" ${output.toTensor[T].nElement()}")
    require(target.toTensor[T].nElement() == negNum + 1,
      s"negNum is $negNum, target's nElement should be ${negNum}, but got" +
        s" ${output.toTensor[T].nElement()}")
    val o = output.toTensor[T].resize(1 + negNum)
    val t = target.toTensor[T].resize(1 + negNum)

    var positiveItem = 0
    var positiveCount = 0
    var i = 1
    while(i <= t.nElement()) {
      if (t.valueAt(i) == 1) {
        positiveItem = i
        positiveCount += 1
      }
      i += 1
    }

    require(positiveItem != 0, s"${format()}: no positive item.")
    require(positiveCount == 1, s"${format()}: too many positive items, excepted 1," +
      s" but got $positiveCount")

    val ndcg = calNDCG(positiveItem, o, k)

    new ContiguousResult(ndcg, 1, s"NDCG")
  }

  // compute NDCG
  private def calNDCG(index: Int, o: Tensor[T], k: Int): Float = {
    var ranking = 1
    var i = 1
    val precision = ev.toType[Float](o.valueAt(index))
    while (i < o.nElement() && ranking <= k) {
      if (ev.toType[Float](o.valueAt(i)) > precision) {
        ranking += 1
      }
      i += 1
    }

    if(ranking <= k) {
      (math.log(2) / math.log(ranking + 1)).toFloat
    } else {
      0
    }
  }

  override def format(): String = "NDCG"
}

/**
 * Use loss as a validation result
 *
 * @param loss loss calculated by forward function
 * @param count recording the times of calculating loss
 */
class LossResult(private var loss: Float, private var count: Int)
  extends ContiguousResult(loss, count, name = "Loss")

/**
 * A generic result type who's data is contiguous float.
 *
 * @param contiResult loss calculated by forward function
 * @param count recording the times of calculating loss
 * @param name name of the result
 */
class ContiguousResult(
    private var contiResult: Float,
    private var count: Int,
    private val name: String)
  extends ValidationResult {

  override def result(): (Float, Int) = (contiResult.toFloat / count, count)

  // scalastyle:off methodName
  override def +(other: ValidationResult): ValidationResult = {
    val otherResult = other.asInstanceOf[ContiguousResult]
    this.contiResult += otherResult.contiResult
    this.count += otherResult.count
    this
  }

  // scalastyle:on methodName

  override protected def format(): String = {
    s"($name: $contiResult, count: $count, Average $name: ${contiResult.toFloat / count})"
  }

  override def equals(obj: Any): Boolean = {
    if (obj == null) {
      return false
    }
    if (!obj.isInstanceOf[ContiguousResult]) {
      return false
    }
    val other = obj.asInstanceOf[ContiguousResult]
    if (this.eq(other)) {
      return true
    }
    this.contiResult == other.contiResult && this.count == other.count
  }

  override def hashCode(): Int = {
    val seed = 37
    var hash = 1
    hash = hash * seed + this.contiResult.toInt
    hash = hash * seed + this.count
    hash
  }
}

/**
 * This evaluation method is calculate loss of output with respect to target
 *
 * @param criterion criterion method for evaluation
 * The default criterion is [[ClassNLLCriterion]]
 */
class Loss[@specialized(Float, Double)T: ClassTag](
 var criterion: Criterion[T] = null)
(implicit ev: TensorNumeric[T]) extends ValidationMethod[T] {
  if (criterion == null) criterion = ClassNLLCriterion[T]()
  override def apply(output: Activity, target: Activity): LossResult = {
    val _target = target.asInstanceOf[Tensor[T]]
    val _output = if (output.toTensor[T].nDimension() != 1 &&
      output.toTensor[T].size().head != _target.size().head) {
      output.toTensor[T].narrow(1, 1, _target.size().head)
    } else {
      output.toTensor[T]
    }
    val loss = ev.toType[Float](criterion.forward(_output, _target))
    val count = _target.size().head

    new LossResult(loss * count, count)
  }

  override def format(): String = "Loss"
}

/**
 * This evaluation method is calculate mean absolute error of output with respect to target
 *
 */
class MAE[@specialized(Float, Double)T: ClassTag]()
(implicit ev: TensorNumeric[T]) extends ValidationMethod[T] {
  private val criterion = AbsCriterion[T]()
  override def apply(output: Activity, target: Activity): LossResult = {
    val _output = output.asInstanceOf[Tensor[T]]
    val (max_prob, max_index) = _output.max(2)
    val _target = target.asInstanceOf[Tensor[T]]
    val loss = ev.toType[Float](criterion.forward(max_index, _target))
    val count = 1

    new LossResult(loss, count)
  }

  override def format(): String = "MAE"
}
