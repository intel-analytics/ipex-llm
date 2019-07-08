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

/**
This validation method is used to obtain the performance in therms of recall:
  tp / (tp + fn). This class requires a input class as parameter.
  */
class Recall[T: ClassTag](_class: T)
 (implicit ev: TensorNumeric[T])
  extends ValidationMethod[T] {

  var tpByClass : Int = 0
  var lcByClass : Int = 0

  override def apply(output: Activity, target: Activity):
  ValidationResult = {

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
      }).map(_target, (b, a) => {
        if (a == b) {
          if (a == _class) {
            tpByClass += 1
          }
        } else {
          if (b == _class) {
            lcByClass += 1
          }
        }
        a
      })
    } else if (_output.dim == 1) {
      require(_target.size(1) == 1)
      (if (_output.size(1) == 1) {
        _output.apply1(x => if (ev.isGreater(ev.fromType(0.5), x)) ev.zero else ev.one)
      } else {
        _output.max(1)._2
      }).map(_target, (b, a) => {
        if (a == b) {
          if (a == _class) {
            tpByClass += 1
          }
        } else {
          // Each key corresponds to the false predictions that have been  made
          if (b == _class) {
            lcByClass += 1
          }
        }
        a
      })
    } else {
      throw new IllegalArgumentException
    }

    new RecallResult(_class, tpByClass, lcByClass)
  }

  override def format(): String = "RECALL"
}

class RecallResult[T: ClassTag]
(_class: T, private var tpByClass: Int, private var lcByClass: Int)
(implicit ev: TensorNumeric[T])
  extends ValidationResult {

  override def result(): (Float, Int) = {
    if (tpByClass != 0.0f || lcByClass != 0.0f) {
      (tpByClass.toFloat / (tpByClass + lcByClass), ev.toType[Int](_class))
    }
    else {
      (0.0f, ev.toType[Int](_class))
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
    s"Recall for class ${_class}: ${result()._1}"
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
class Precision[T: ClassTag](_class: T)
  (implicit ev: TensorNumeric[T])
  extends ValidationMethod[T] {

  var tpByClass : Int = 0
  var fpByClass : Int = 0

  override def apply(output: Activity, target: Activity):

  ValidationResult = {
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
      }).map(_target, (b, a) => {
        if (a == _class) {
          if (a == b) {
            tpByClass += 1
          }
          else {
            fpByClass += 1
          }
        }
        a
      })
    } else if (_output.dim == 1) {
      require(_target.size(1) == 1)
      (if (_output.size(1) == 1) {
        _output.apply1(x => if (ev.isGreater(ev.fromType(0.5), x)) ev.zero else ev.one)
      } else {
        _output.max(1)._2
      }).map(_target, (b, a) => {
        if (a == _class) {
          if (a == b) {
            tpByClass += 1
          }
          else {
            fpByClass += 1
          }
        }
        a
      })
    } else {
      throw new IllegalArgumentException
    }

    new PrecisionResult(_class, tpByClass, fpByClass)
  }

  override def format(): String = "PRECISION"
}

class PrecisionResult[T: ClassTag](_class: T,
                                   private var tpByClass: Int,
                                   private var fpByClass: Int)
                                  (implicit ev: TensorNumeric[T])
  extends ValidationResult {

  override def result(): (Float, Int) = {
    if (tpByClass != 0 || fpByClass != 0) {
      (tpByClass.toFloat / (tpByClass + fpByClass), ev.toType[Int](_class))
    }
    else {
      (0.0f, ev.toType[Int](_class))
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

  override protected def format(): String = s"Precision for class ${_class}: ${result()._1}"
}

class F1Score[T: ClassTag](_class: T)
                          (implicit ev: TensorNumeric[T])
  extends ValidationMethod[T] {

  var tpByClass : Int = 0
  var fpByClass : Int = 0
  val lcByClass : Int = 0

  override def apply(output: Activity, target: Activity) :
  ValidationResult = {
    val pr = new Precision[T](_class)
    val rc = new Recall[T](_class)
    pr.apply(output, target)
    rc.apply(output, target)
    new F1ScoreResult[T](_class, pr.tpByClass, pr.fpByClass, rc.lcByClass)
  }

  override protected def format(): String = "F1Score"
}

class F1ScoreResult[T: ClassTag](_class: T, private var tpByClass: Int,
  private var fpByClass: Int, private var lcByClass: Int)
  (implicit ev: TensorNumeric[T]) extends ValidationResult {
  override def result(): (Float, Int) = {

    val precision = if (tpByClass != 0 || fpByClass != 0) {
      (tpByClass.toFloat / (tpByClass + fpByClass), tpByClass)
    }
    else {
      (0.0f, tpByClass)
    }

    val recall = {
      if (tpByClass != 0.0f || lcByClass != 0.0f) {
        (tpByClass.toFloat / (tpByClass + lcByClass), ev.toType[Int](_class))
      }
      else {
        (0.0f, ev.toType[Int](_class))
      }
    }

    // harmonic mean
    ((2 * (precision._1 * recall._1)) / ( precision._1 + recall._1), ev.toType[Int](_class))
  }

  override def +(other: ValidationResult): ValidationResult = {
    val otherResult = other.asInstanceOf[F1ScoreResult[T]]
    this.lcByClass += otherResult.lcByClass
    this.tpByClass += otherResult.tpByClass
    this.fpByClass += otherResult.fpByClass
    this
  }

  override def equals(obj: Any): Boolean = {
    if (obj == null) {
      return false
    }
    if (!obj.isInstanceOf[F1ScoreResult[T]]) {
      return false
    }
    val other = obj.asInstanceOf[F1ScoreResult[T]]
    if (this.eq(other)) {
      return true
    }
    this.tpByClass == other.tpByClass && this.fpByClass == other.fpByClass &&
      this.lcByClass == other.lcByClass
  }

  override def hashCode(): Int = {
    val seed = 37
    var hash = 1
    hash = hash * seed + this.fpByClass
    hash = hash * seed + this.tpByClass
    hash = hash * seed + this.lcByClass
    hash
  }

  override protected def format(): String = s"F1Score for class ${_class}: ${result()._1}"
}