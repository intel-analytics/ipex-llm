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
import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.nn.AbsCriterion
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.commons.lang3.SerializationUtils

import scala.collection.immutable.ListMap
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
        _output.apply1(x => if (ev.isGreater(ev.fromType(0.5), x)) ev.zero else ev.one)
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
        _output.apply1(x => if (ev.isGreater(ev.fromType(0.5), x)) ev.zero else ev.one)
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
        _output.apply1(x => if (ev.isGreater(ev.fromType(0.5), x)) ev.zero else ev.one)
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
        _output.apply1(x => if (ev.isGreater(ev.fromType(0.5), x)) ev.zero else ev.one)
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
 * Caculate the percentage that target in output's top5 probability indexes
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
    val count = 1

    new LossResult(loss, count)
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

object TensorValidationOps {
  /*
  This function converts 2D tensors
   */
  def checkTensorDims[T: ClassTag](output: Activity, target: Activity)
   (implicit ev: TensorNumeric[T]): Tensor[T] = {
    val _target = target.asInstanceOf[Tensor[T]]
    if (output.toTensor[T].nDimension() != 1 &&
      output.toTensor[T].size().head != _target.size().head) {
      output.toTensor[T].narrow(1, 1, _target.size().head)
    } else {
      output.toTensor[T]
    }
  }

  def transformOutput[T: ClassTag, A](output: Tensor[T], target: Tensor[T])
   (implicit ev: TensorNumeric[T]): Tensor[T] = {
    val nDim = output.dim()
    nDim match {
      case 2 =>
        if (output.size(2) == 1) {
          output.apply1(x => if (ev.isGreater(ev.fromType(0.5), x)) ev.zero else ev.one)
        } else {
          output.max(2)._2.squeeze()
        }
      case 1 =>
        require(target.size(1) == 1)
        if (output.size(1) == 1) {
          output.apply1(x => if (ev.isGreater(ev.fromType(0.5), x)) ev.zero else ev.one)
        } else {
          output.max(1)._2
        }
      case _ =>
        throw new IllegalArgumentException
    }
  }
}

import TensorValidationOps._

/**
This validation method is used to obtain the performance in therms of recall:
  tp / (tp + fn). This class requires an input class as parameter.
  */
class Recall[T: ClassTag](classId: T)
 (implicit ev: TensorNumeric[T])
  extends ValidationMethod[T] {

  override def apply(output: Activity, target: Activity):
  ValidationResult = {

    var tpByClass : Int = 0
    var lcByClass : Int = 0

    val _target = target.asInstanceOf[Tensor[T]]
    val _output = checkTensorDims(output, target)

    transformOutput(_output, _target).map(_target, (b, a) => {
      // bs are the predictions
      if (b == classId) {
        if (a == b) {
          tpByClass += 1
        }
      }
      if(a == classId) {
        if (a != b) {
          lcByClass += 1
        }
      }
      a
    })

    new RecallResult(classId, tpByClass, lcByClass)
  }

  override def format(): String = "RECALL"
}

class RecallResult[T: ClassTag]
 (classId: T, private var tpByClass: Int, private var lcByClass: Int)
 (implicit ev: TensorNumeric[T]) extends ValidationResult {

  override def result(): (Float, Int) = {
    if (tpByClass != 0.0f || lcByClass != 0.0f) {
      (tpByClass.toFloat / (tpByClass + lcByClass), ev.toType[Int](classId))
    }
    else {
      (0.0f, ev.toType[Int](classId))
    }
  } // Apply the recall formula TP /(TP + FN)

  // scalastyle:off methodName
  override def +(other: ValidationResult): ValidationResult = {
    val otherResult = other.asInstanceOf[RecallResult[T]]
    this.lcByClass += otherResult.lcByClass
    this.tpByClass += otherResult.tpByClass
    this
  }

  // scalastyle:on methodName
  override protected def format(): String = {
    s"Recall for class ${classId}: ${result()._1}." +
    s"True Predictions: ${tpByClass}. Incorrect predictions in other classes: ${lcByClass}"
  }

  override def equals(obj: Any): Boolean = {
    if (obj == null) {
      return false
    }
    if (!obj.isInstanceOf[RecallResult[T]]) {
      return false
    }
    val other = obj.asInstanceOf[RecallResult[T]]
    if (this.eq(other)) {
      return true
    }
    this.tpByClass == other.tpByClass && this.lcByClass == other.lcByClass
  }

  override def hashCode(): Int = {
    val seed = 37
    var hash = 1
    hash = hash * seed + this.lcByClass
    hash = hash * seed + this.tpByClass
    hash
  }
}

/**
  This validation method is used to obtain the performance in therms of precision:
  tp / (tp + fp). This class requires an input parameter which is the class
  for which we need to obtain the precision.
  */
class Precision[T: ClassTag](classId: T)
  (implicit ev: TensorNumeric[T])
  extends ValidationMethod[T] {

  override def apply(output: Activity, target: Activity):

  ValidationResult = {

    var tpByClass : Int = 0
    var fpByClass : Int = 0

    val _target = target.asInstanceOf[Tensor[T]]
    val _output = checkTensorDims(output, target)

    transformOutput(_output, _target).map(_target, (b, a) => {
      // bs are the predictions
      if (b == classId) {
        if (a == b) {
          tpByClass += 1
        }
        else {
          fpByClass += 1
        }
      }
      a
    })

    new PrecisionResult(classId, tpByClass, fpByClass)
  }

  override def format(): String = "PRECISION"
}

class PrecisionResult[T: ClassTag](classId: T,
  private var tpByClass: Int, private var fpByClass: Int)
  (implicit ev: TensorNumeric[T])
  extends ValidationResult {

  override def result(): (Float, Int) = {
    if (tpByClass != 0 || fpByClass != 0) {
      (tpByClass.toFloat / (tpByClass + fpByClass), ev.toType[Int](classId))
    }
    else {
      (0.0f, ev.toType[Int](classId))
    }
  }

  // scalastyle:off methodName
  override def +(other: ValidationResult): ValidationResult = {
    val otherResult = other.asInstanceOf[PrecisionResult[T]]
    this.fpByClass = otherResult.fpByClass
    this.tpByClass = otherResult.tpByClass
    this
  }

  override def equals(obj: Any): Boolean = {
    if (obj == null) {
      return false
    }
    if (!obj.isInstanceOf[PrecisionResult[T]]) {
      return false
    }
    val other = obj.asInstanceOf[PrecisionResult[T]]
    if (this.eq(other)) {
      return true
    }
    this.tpByClass == other.tpByClass && this.fpByClass == other.fpByClass
  }

  override def hashCode(): Int = {
    val seed = 37
    var hash = 1
    hash = hash * seed + this.fpByClass
    hash = hash * seed + this.tpByClass
    hash
  }

  override protected def format(): String = s"Precision for class ${classId}: " +
    s"${result()._1}. True Predictions: ${tpByClass}. False Predictions: ${fpByClass}"
}


// Available average methods. Thin about including
// weighted macro average.

sealed trait AverageMethod
case object MicroAverage extends AverageMethod
case object MacroAverage extends AverageMethod

class F1ScoreAvg[T: ClassTag](average: AverageMethod)
  (implicit ev: TensorNumeric[T], order: Ordering[T])
  extends ValidationMethod[T] {

  // Using in macro case

  override def apply(output: Activity, target: Activity) :
  ValidationResult = {

    // Using in micro case
    var tpByClass : Map[T, Int] = Map()
    var fpByClass : Map[T, Int] = Map()
    var lcByClass : Map[T, Int] = Map()

    val _target = target.asInstanceOf[Tensor[T]]
    val _output = checkTensorDims(output, target)

    transformOutput(_output, _target).map(_target, (b, a) => {
      if (a == b) {
        tpByClass = tpByClass + (b -> (tpByClass.getOrElse(b, 0) + 1))
      }
      else {
        // Check if there is no pair for true positives
        if (!tpByClass.get(b).isDefined) {
          tpByClass = tpByClass + (b -> 0)
        }

        fpByClass = fpByClass + (b -> (fpByClass.getOrElse(b, 0) + 1))
        lcByClass = lcByClass + (a -> (lcByClass.getOrElse(a, 0) + 1))
      }
      a
    })

    new F1ScoreAvgResult(average, tpByClass, fpByClass, lcByClass)
  }

  override protected def format(): String = "F1ScoreAvg"
}

class F1ScoreAvgResult[T: ClassTag](average: AverageMethod, private var tpByClass: Map[T, Int],
  private var fpByClass: Map[T, Int],
  private var lcByClass: Map[T, Int])
 (implicit ev: TensorNumeric[T], order: Ordering[T]) extends ValidationResult {

  private var precisionValue : Float = 0.0F
  private var recallValue : Float = 0.0F
  private var f1score : Float = 0.0F

  override def result(): (Float, Int) = {
    /*
      Using micro or macro average
     */
    average match {
      case MicroAverage =>
        // Sum tp and fp
        val tps = tpByClass.values.sum

        precisionValue = {
          val fps = fpByClass.values.sum
          tps.toFloat / (tps + fps)
        }

        recallValue = {
          val lcs = lcByClass.values.sum
          tps.toFloat / (tps + lcs)
        }

        ((2 * (precisionValue * recallValue)) / ( precisionValue + recallValue), 0)

      case MacroAverage =>
        // Macro weight
        val tupledMetrics =
          (tpByClass.toSeq.sortBy(_._1),
            fpByClass.toSeq.sortBy(_._1), lcByClass.toSeq.sortBy(_._1)).zipped

        val results = tupledMetrics.map {
          case ((cls, tp), (cls2, fp), (cls3, lc)) =>
            val precision = tp.toFloat / (tp + fp)
            val recall = tp.toFloat / (lc + tp)
            (cls, (precision, recall))
        }


        val metricsList = results.map { case (classId, (precision, recall)) =>
          val f1scorePerClass = {
            if (precision != 0 || recall != 0) {
              (2 * (precision * recall)) / ( precision + recall)
            } else {
              0
            }
          }
          (f1scorePerClass, precision, recall)
        }

        f1score = metricsList.map(_._1).sum / results.size
        precisionValue = metricsList.map(_._2).sum / results.size
        recallValue = metricsList.map(_._3).sum / results.size

        (f1score, 0)
    }
  }

  override def +(other: ValidationResult): ValidationResult = {
    val otherResult = other.asInstanceOf[F1ScoreAvgResult[T]]
    this.lcByClass ++= otherResult.lcByClass
    this.tpByClass ++= otherResult.tpByClass
    this.fpByClass ++= otherResult.fpByClass
    this
  }

  override def equals(obj: Any): Boolean = {
    tpByClass.eq(fpByClass) && fpByClass.eq(lcByClass)
  }

  override def hashCode(): Int = {
    val seed = 37
    var hash = 1
    hash = hash * seed + this.fpByClass.hashCode()
    hash = hash * seed + this.tpByClass.hashCode()
    hash = hash * seed + this.lcByClass.hashCode()
    hash
  }

  override protected def format(): String = {
    val (v, _) = result()
    s"Results for type ${average}: " +
      s"Precision ${precisionValue}. Recall ${recallValue}. F1 ${v}"
  }
}