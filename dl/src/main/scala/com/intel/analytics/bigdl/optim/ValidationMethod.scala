/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.tensor.Tensor

trait ValidationMethod[T] {
  def apply(output: Tensor[T], target: Tensor[T]): ValidationResult

  def format(): String

  override def toString(): String = format()
}

trait ValidationResult {

  // scalastyle:off methodName
  def ++(other: ValidationResult): ValidationResult

  // scalastyle:on methodName

  protected def format(): String

  override def toString(): String = format()
}

class AccuracyResult(private var correct: Int, private var count: Int)
  extends ValidationResult {

  // scalastyle:off methodName
  override def ++(other: ValidationResult): ValidationResult = {
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

class Top1Accuracy[T] extends ValidationMethod[T] {
  override def apply(output: Tensor[T], target: Tensor[T]): ValidationResult = {
    var correct = 0
    var count = 0

    if (output.dim() == 2) {
      output.max(2)._2.squeeze().map(target, (a, b) => {
        if (a == b) {
          correct += 1
        }
        a
      })
      count += output.size(1)
    } else if (output.dim == 1) {
      require(target.size(1) == 1)
      output.max(1)._2.map(target, (a, b) => {
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

  override def format(): String = "top1 accuracy"
}

class Top5Accuracy[T] extends ValidationMethod[T] {
  override def apply(output: Tensor[T], target: Tensor[T]): AccuracyResult = {
    var correct = 0
    var count = 0
    if (output.dim() == 2) {
      val indices = output.topk(5, 2, false)._2
      var i = 1
      while (i <= output.size(1)) {
        if (indices.valueAt(i, 1) == target.valueAt(i)
          || indices.valueAt(i, 2) == target.valueAt(i)
          || indices.valueAt(i, 3) == target.valueAt(i)
          || indices.valueAt(i, 4) == target.valueAt(i)
          || indices.valueAt(i, 5) == target.valueAt(i)) {
          correct += 1
        }
        i += 1
      }
      count += output.size(1)
    } else if (output.dim == 1) {
      require(target.size(1) == 1)
      val indices = output.topk(5, 1, false)._2
      if (indices.valueAt(1) == target.valueAt(1) || indices.valueAt(2) == target.valueAt(1)
        || indices.valueAt(3) == target.valueAt(1) || indices.valueAt(4) == target.valueAt(1)
        || indices.valueAt(5) == target.valueAt(1)) {
        correct += 1
      }
      count += 1
    } else {
      throw new IllegalArgumentException
    }

    new AccuracyResult(correct, count)
  }

  override def format(): String = "top5 accuracy"
}
