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

package com.intel.analytics.sparkdl.optim

import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric._
import org.apache.spark.SparkContext
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class ShuffleBatchDataSetSpec extends FlatSpec with Matchers with BeforeAndAfter {
  var sc: SparkContext = null

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  "fetch one batch with same size of stack" should "be correct" in {
    sc = new SparkContext("local[4]", "ShuffleBatchDataSetSpec")
    val rdd = sc.parallelize(Array(1, 2, 3, 4), 4).mapPartitions(iter => Iterator.range(0, 8))
    val dataset = new ShuffleBatchDataSet[Int, Double](rdd,
      (seq, input: Tensor[Double], target: Tensor[Double]) => {
        input.resize(Array(seq.size - 1))
        target.resize(Array(1))

        var i = 0
        while (i < seq.size - 1) {
          input.setValue(i + 1, seq(i))
          i += 1
        }
        target.setValue(1, seq(i))
        (input, target)
      }, 4, 4, 1)

    require(!dataset.epochFinished())

    dataset.fetch().map(batch => {
      require(batch.hasNext)
      val b = batch.next()
      require(!batch.hasNext)
      require(b._1.size(1) == 3)
      require(b._1.valueAt(1) == 0.0, b._1.valueAt(1))
      require(b._1.valueAt(2) == 1.0, b._1.valueAt(2))
      require(b._1.valueAt(3) == 2.0, b._1.valueAt(3))
      require(b._2.size(1) == 1)
      require(b._2.valueAt(1) == 3.0, b._2.valueAt(1))
    }).count()

    dataset.fetch().map(batch => {
      require(batch.hasNext)
      val b = batch.next()
      require(!batch.hasNext)
      require(b._1.size(1) == 3)
      require(b._1.valueAt(1) == 4.0, b._1.valueAt(1))
      require(b._1.valueAt(2) == 5.0, b._1.valueAt(2))
      require(b._1.valueAt(3) == 6.0, b._1.valueAt(3))
      require(b._2.size(1) == 1)
      require(b._2.valueAt(1) == 7.0, b._1.valueAt(4))
    }).count()

    require(dataset.epochFinished())

    dataset.fetch().map(batch => {
      require(batch.hasNext)
      val b = batch.next()
      require(!batch.hasNext)
      require(b._1.size(1) == 3)
      require(b._1.valueAt(1) == 0.0, b._1.valueAt(1))
      require(b._1.valueAt(2) == 1.0, b._1.valueAt(2))
      require(b._1.valueAt(3) == 2.0, b._1.valueAt(3))
      require(b._2.size(1) == 1)
      require(b._2.valueAt(1) == 3.0, b._2.valueAt(1))
    }).count()

    dataset.fetch().map(batch => {
      require(batch.hasNext)
      val b = batch.next()
      require(!batch.hasNext)
      require(b._1.size(1) == 3)
      require(b._1.valueAt(1) == 4.0, b._1.valueAt(1))
      require(b._1.valueAt(2) == 5.0, b._1.valueAt(2))
      require(b._1.valueAt(3) == 6.0, b._1.valueAt(3))
      require(b._2.size(1) == 1)
      require(b._2.valueAt(1) == 7.0, b._1.valueAt(4))
    }).count()
  }

  "fetch one batch with different size of stack" should "be correct" in {
    sc = new SparkContext("local[4]", "ShuffleBatchDataSetSpec")
    val rdd = sc.parallelize(Array(1, 2, 3, 4), 4).
      mapPartitions(iter => Iterator.range(0, 8).map(i => (i, i)))
    val dataset = new ShuffleBatchDataSet[(Int, Int), Double](rdd,
      (seq, input: Tensor[Double], target: Tensor[Double]) => {
        input.resize(Array(seq.size, 1))
        target.resize(Array(seq.size))

        var i = 0
        while (i < seq.size) {
          input.setValue(i + 1, 1, seq(i)._1)
          target.setValue(i + 1, seq(i)._2)
          i += 1
        }
        (input, target)
      }, 3, 4, 1)

    require(!dataset.epochFinished())

    dataset.fetch().map(batch => {
      require(batch.hasNext)
      val b1 = batch.next()
      require(b1._1.size(1) == 3)
      require(b1._1.valueAt(1, 1) == 0.0, b1._1.valueAt(1, 1))
      require(b1._1.valueAt(2, 1) == 1.0, b1._1.valueAt(1, 1))
      require(b1._1.valueAt(3, 1) == 2.0, b1._1.valueAt(1, 1))
      require(b1._2.size(1) == 3)
      require(b1._2.valueAt(1) == 0.0, b1._2.valueAt(1, 1))
      require(b1._2.valueAt(2) == 1.0, b1._2.valueAt(1, 1))
      require(b1._2.valueAt(3) == 2.0, b1._2.valueAt(1, 1))

      val b2 = batch.next()
      require(!batch.hasNext)
      require(b2._1.size(1) == 1)
      require(b2._1.valueAt(1, 1) == 3.0, b2._1.valueAt(1, 1))
      require(b2._2.size(1) == 1)
      require(b2._2.valueAt(1) == 3.0, b2._2.valueAt(1, 1))
    }).count()

    dataset.fetch().map(batch => {
      require(batch.hasNext)
      val b1 = batch.next()
      require(b1._1.size(1) == 3)
      require(b1._1.valueAt(1, 1) == 4.0, b1._1.valueAt(1, 1))
      require(b1._1.valueAt(2, 1) == 5.0, b1._1.valueAt(1, 1))
      require(b1._1.valueAt(3, 1) == 6.0, b1._1.valueAt(1, 1))
      require(b1._2.size(1) == 3)
      require(b1._2.valueAt(1) == 4.0, b1._2.valueAt(1, 1))
      require(b1._2.valueAt(2) == 5.0, b1._2.valueAt(1, 1))
      require(b1._2.valueAt(3) == 6.0, b1._2.valueAt(1, 1))

      val b2 = batch.next()
      require(!batch.hasNext)
      require(b2._1.size(1) == 1)
      require(b2._1.valueAt(1, 1) == 7.0, b2._1.valueAt(1, 1))
      require(b2._2.size(1) == 1)
      require(b2._2.valueAt(1) == 7.0, b2._2.valueAt(1, 1))
    }).count()

    require(dataset.epochFinished())

    dataset.fetch().map(batch => {
      require(batch.hasNext)
      val b1 = batch.next()
      require(b1._1.size(1) == 3)
      require(b1._1.valueAt(1, 1) == 0.0, b1._1.valueAt(1, 1))
      require(b1._1.valueAt(2, 1) == 1.0, b1._1.valueAt(1, 1))
      require(b1._1.valueAt(3, 1) == 2.0, b1._1.valueAt(1, 1))
      require(b1._2.size(1) == 3)
      require(b1._2.valueAt(1) == 0.0, b1._2.valueAt(1, 1))
      require(b1._2.valueAt(2) == 1.0, b1._2.valueAt(1, 1))
      require(b1._2.valueAt(3) == 2.0, b1._2.valueAt(1, 1))

      val b2 = batch.next()
      require(!batch.hasNext)
      require(b2._1.size(1) == 1)
      require(b2._1.valueAt(1, 1) == 3.0, b2._1.valueAt(1, 1))
      require(b2._2.size(1) == 1)
      require(b2._2.valueAt(1) == 3.0, b2._2.valueAt(1, 1))
    }).count()

    dataset.fetch().map(batch => {
      require(batch.hasNext)
      val b1 = batch.next()
      require(b1._1.size(1) == 3)
      require(b1._1.valueAt(1, 1) == 4.0, b1._1.valueAt(1, 1))
      require(b1._1.valueAt(2, 1) == 5.0, b1._1.valueAt(1, 1))
      require(b1._1.valueAt(3, 1) == 6.0, b1._1.valueAt(1, 1))
      require(b1._2.size(1) == 3)
      require(b1._2.valueAt(1) == 4.0, b1._2.valueAt(1, 1))
      require(b1._2.valueAt(2) == 5.0, b1._2.valueAt(1, 1))
      require(b1._2.valueAt(3) == 6.0, b1._2.valueAt(1, 1))

      val b2 = batch.next()
      require(!batch.hasNext)
      require(b2._1.size(1) == 1)
      require(b2._1.valueAt(1, 1) == 7.0, b2._1.valueAt(1, 1))
      require(b2._2.size(1) == 1)
      require(b2._2.valueAt(1) == 7.0, b2._2.valueAt(1, 1))
    }).count()
  }

  it should "be correct for multithread dataset" in {
    sc = new SparkContext("local[1]", "ShuffleBatchDataSetSpec")
    val rdd = sc.parallelize(Array(1, 2, 3, 4), 4).
      mapPartitions(iter => Iterator.range(0, 8).map(i => (i, i)))
    val dataset = new MultiThreadShuffleBatchDataSet[(Int, Int), Double](rdd,
      (seq, input: Tensor[Double], target: Tensor[Double]) => {
        input.resize(Array(seq.size, 1))
        target.resize(Array(seq.size))

        var i = 0
        while (i < seq.size) {
          input.setValue(i + 1, 1, seq(i)._1)
          target.setValue(i + 1, seq(i)._2)
          i += 1
        }
        (input, target)
      }, 3, 4, 1)

    require(!dataset.epochFinished())

    dataset.fetch().map(batch => {
      require(batch.hasNext)
      val b1 = batch.next()
      require(b1._1.size(1) == 3)
      require(b1._1.valueAt(1, 1) == 0.0, b1._1.valueAt(1, 1))
      require(b1._1.valueAt(2, 1) == 1.0, b1._1.valueAt(1, 1))
      require(b1._1.valueAt(3, 1) == 2.0, b1._1.valueAt(1, 1))
      require(b1._2.size(1) == 3)
      require(b1._2.valueAt(1) == 0.0, b1._2.valueAt(1, 1))
      require(b1._2.valueAt(2) == 1.0, b1._2.valueAt(1, 1))
      require(b1._2.valueAt(3) == 2.0, b1._2.valueAt(1, 1))

      val b2 = batch.next()
      require(!batch.hasNext)
      require(b2._1.size(1) == 1)
      require(b2._1.valueAt(1, 1) == 3.0, b2._1.valueAt(1, 1))
      require(b2._2.size(1) == 1)
      require(b2._2.valueAt(1) == 3.0, b2._2.valueAt(1, 1))
    }).count()

    dataset.fetch().map(batch => {
      require(batch.hasNext)
      val b1 = batch.next()
      require(b1._1.size(1) == 3)
      require(b1._1.valueAt(1, 1) == 4.0, b1._1.valueAt(1, 1))
      require(b1._1.valueAt(2, 1) == 5.0, b1._1.valueAt(1, 1))
      require(b1._1.valueAt(3, 1) == 6.0, b1._1.valueAt(1, 1))
      require(b1._2.size(1) == 3)
      require(b1._2.valueAt(1) == 4.0, b1._2.valueAt(1, 1))
      require(b1._2.valueAt(2) == 5.0, b1._2.valueAt(1, 1))
      require(b1._2.valueAt(3) == 6.0, b1._2.valueAt(1, 1))

      val b2 = batch.next()
      require(!batch.hasNext)
      require(b2._1.size(1) == 1)
      require(b2._1.valueAt(1, 1) == 7.0, b2._1.valueAt(1, 1))
      require(b2._2.size(1) == 1)
      require(b2._2.valueAt(1) == 7.0, b2._2.valueAt(1, 1))
    }).count()

    require(dataset.epochFinished())

    dataset.fetch().map(batch => {
      require(batch.hasNext)
      val b1 = batch.next()
      require(b1._1.size(1) == 3)
      require(b1._1.valueAt(1, 1) == 0.0, b1._1.valueAt(1, 1))
      require(b1._1.valueAt(2, 1) == 1.0, b1._1.valueAt(1, 1))
      require(b1._1.valueAt(3, 1) == 2.0, b1._1.valueAt(1, 1))
      require(b1._2.size(1) == 3)
      require(b1._2.valueAt(1) == 0.0, b1._2.valueAt(1, 1))
      require(b1._2.valueAt(2) == 1.0, b1._2.valueAt(1, 1))
      require(b1._2.valueAt(3) == 2.0, b1._2.valueAt(1, 1))

      val b2 = batch.next()
      require(!batch.hasNext)
      require(b2._1.size(1) == 1)
      require(b2._1.valueAt(1, 1) == 3.0, b2._1.valueAt(1, 1))
      require(b2._2.size(1) == 1)
      require(b2._2.valueAt(1) == 3.0, b2._2.valueAt(1, 1))
    }).count()

    dataset.fetch().map(batch => {
      require(batch.hasNext)
      val b1 = batch.next()
      require(b1._1.size(1) == 3)
      require(b1._1.valueAt(1, 1) == 4.0, b1._1.valueAt(1, 1))
      require(b1._1.valueAt(2, 1) == 5.0, b1._1.valueAt(1, 1))
      require(b1._1.valueAt(3, 1) == 6.0, b1._1.valueAt(1, 1))
      require(b1._2.size(1) == 3)
      require(b1._2.valueAt(1) == 4.0, b1._2.valueAt(1, 1))
      require(b1._2.valueAt(2) == 5.0, b1._2.valueAt(1, 1))
      require(b1._2.valueAt(3) == 6.0, b1._2.valueAt(1, 1))

      val b2 = batch.next()
      require(!batch.hasNext)
      require(b2._1.size(1) == 1)
      require(b2._1.valueAt(1, 1) == 7.0, b2._1.valueAt(1, 1))
      require(b2._2.size(1) == 1)
      require(b2._2.valueAt(1) == 7.0, b2._2.valueAt(1, 1))
    }).count()
  }

  "fetch two batch with same size of stack" should "be correct" in {
    sc = new SparkContext("local[4]", "ShuffleBatchDataSetSpec")
    val rdd = sc.parallelize(Array(1, 2, 3, 4), 4).
      mapPartitions(iter => Iterator.range(0, 8).map(i => (i, i)))
    val dataset = new ShuffleBatchDataSet[(Int, Int), Double](rdd,
      (seq, input: Tensor[Double], target: Tensor[Double]) => {
        input.resize(Array(seq.size, 1))
        target.resize(Array(seq.size))

        var i = 0
        while (i < seq.size) {
          input.setValue(i + 1, 1, seq(i)._1)
          target.setValue(i + 1, seq(i)._2)
          i += 1
        }
        (input, target)
      }, 4, 4, 2)

    require(!dataset.epochFinished())

    dataset.fetch().mapPartitions(batchIter => {
      require(batchIter.hasNext)
      var batch = batchIter.next()
      require(batch.hasNext)
      var b = batch.next()
      require(!batch.hasNext)
      require(b._1.size(1) == 4)
      require(b._1.valueAt(1, 1) == 0.0, b._1.valueAt(1, 1))
      require(b._1.valueAt(2, 1) == 1.0, b._1.valueAt(2, 1))
      require(b._1.valueAt(3, 1) == 2.0, b._1.valueAt(3, 1))
      require(b._1.valueAt(4, 1) == 3.0, b._1.valueAt(4, 1))
      require(b._2.size(1) == 4)
      require(b._2.valueAt(1) == 0.0, b._2.valueAt(1))
      require(b._2.valueAt(2) == 1.0, b._2.valueAt(2))
      require(b._2.valueAt(3) == 2.0, b._2.valueAt(3))
      require(b._2.valueAt(4) == 3.0, b._2.valueAt(4))

      batch = batchIter.next()
      require(!batchIter.hasNext)
      require(batch.hasNext)
      b = batch.next()
      require(!batch.hasNext)
      require(b._1.size(1) == 4)
      require(b._1.valueAt(1, 1) == 4.0, b._1.valueAt(1, 1))
      require(b._1.valueAt(2, 1) == 5.0, b._1.valueAt(2, 1))
      require(b._1.valueAt(3, 1) == 6.0, b._1.valueAt(3, 1))
      require(b._1.valueAt(4, 1) == 7.0, b._1.valueAt(4, 1))
      require(b._2.size(1) == 4)
      require(b._2.valueAt(1) == 4.0, b._2.valueAt(1))
      require(b._2.valueAt(2) == 5.0, b._2.valueAt(2))
      require(b._2.valueAt(3) == 6.0, b._2.valueAt(3))
      require(b._2.valueAt(4) == 7.0, b._2.valueAt(4))

      Iterator.empty
    }).count()

    require(dataset.epochFinished())

    dataset.fetch().mapPartitions(batchIter => {
      require(batchIter.hasNext)
      var batch = batchIter.next()
      require(batch.hasNext)
      var b = batch.next()
      require(!batch.hasNext)
      require(b._1.size(1) == 4)
      require(b._1.valueAt(1, 1) == 0.0, b._1.valueAt(1, 1))
      require(b._1.valueAt(2, 1) == 1.0, b._1.valueAt(2, 1))
      require(b._1.valueAt(3, 1) == 2.0, b._1.valueAt(3, 1))
      require(b._1.valueAt(4, 1) == 3.0, b._1.valueAt(4, 1))
      require(b._2.size(1) == 4)
      require(b._2.valueAt(1) == 0.0, b._2.valueAt(1))
      require(b._2.valueAt(2) == 1.0, b._2.valueAt(2))
      require(b._2.valueAt(3) == 2.0, b._2.valueAt(3))
      require(b._2.valueAt(4) == 3.0, b._2.valueAt(4))

      batch = batchIter.next()
      require(!batchIter.hasNext)
      require(batch.hasNext)
      b = batch.next()
      require(!batch.hasNext)
      require(b._1.size(1) == 4)
      require(b._1.valueAt(1, 1) == 4.0, b._1.valueAt(1, 1))
      require(b._1.valueAt(2, 1) == 5.0, b._1.valueAt(2, 1))
      require(b._1.valueAt(3, 1) == 6.0, b._1.valueAt(3, 1))
      require(b._1.valueAt(4, 1) == 7.0, b._1.valueAt(4, 1))
      require(b._2.size(1) == 4)
      require(b._2.valueAt(1) == 4.0, b._2.valueAt(1))
      require(b._2.valueAt(2) == 5.0, b._2.valueAt(2))
      require(b._2.valueAt(3) == 6.0, b._2.valueAt(3))
      require(b._2.valueAt(4) == 7.0, b._2.valueAt(4))

      Iterator.empty
    }).count()
  }

  "fetch two batch with same size of stack with shuffle" should "be correct" in {
    sc = new SparkContext("local[4]", "ShuffleBatchDataSetSpec")
    val rdd = sc.parallelize(Array(1, 2, 3, 4), 4).
      mapPartitions(iter => Iterator.range(0, 8).map(i => (i, i)))
    val dataset = new ShuffleBatchDataSet[(Int, Int), Double](rdd,
      (seq, input: Tensor[Double], target: Tensor[Double]) => {
        input.resize(Array(seq.size, 1))
        target.resize(Array(seq.size))

        var i = 0
        while (i < seq.size) {
          input.setValue(i + 1, 1, seq(i)._1)
          target.setValue(i + 1, seq(i)._2)
          i += 1
        }
        (input, target)
      }, 4, 4, 2)

    require(!dataset.epochFinished())

    dataset.fetch().mapPartitions(batchIter => {
      require(batchIter.hasNext)
      var batch = batchIter.next()
      require(batch.hasNext)
      var b = batch.next()
      require(!batch.hasNext)
      require(b._1.size(1) == 4)
      require(b._1.valueAt(1, 1) == 0.0, b._1.valueAt(1, 1))
      require(b._1.valueAt(2, 1) == 1.0, b._1.valueAt(2, 1))
      require(b._1.valueAt(3, 1) == 2.0, b._1.valueAt(3, 1))
      require(b._1.valueAt(4, 1) == 3.0, b._1.valueAt(4, 1))
      require(b._2.size(1) == 4)
      require(b._2.valueAt(1) == 0.0, b._2.valueAt(1))
      require(b._2.valueAt(2) == 1.0, b._2.valueAt(2))
      require(b._2.valueAt(3) == 2.0, b._2.valueAt(3))
      require(b._2.valueAt(4) == 3.0, b._2.valueAt(4))

      batch = batchIter.next()
      require(!batchIter.hasNext)
      require(batch.hasNext)
      b = batch.next()
      require(!batch.hasNext)
      require(b._1.size(1) == 4)
      require(b._1.valueAt(1, 1) == 4.0, b._1.valueAt(1, 1))
      require(b._1.valueAt(2, 1) == 5.0, b._1.valueAt(2, 1))
      require(b._1.valueAt(3, 1) == 6.0, b._1.valueAt(3, 1))
      require(b._1.valueAt(4, 1) == 7.0, b._1.valueAt(4, 1))
      require(b._2.size(1) == 4)
      require(b._2.valueAt(1) == 4.0, b._2.valueAt(1))
      require(b._2.valueAt(2) == 5.0, b._2.valueAt(2))
      require(b._2.valueAt(3) == 6.0, b._2.valueAt(3))
      require(b._2.valueAt(4) == 7.0, b._2.valueAt(4))

      Iterator.empty
    }).count()

    require(dataset.epochFinished())
    dataset.reset()
    require(!dataset.epochFinished())

    dataset.fetch().mapPartitions(batchIter => {
      val data = new Array[Double](8)
      require(batchIter.hasNext)
      var batch = batchIter.next()
      require(batch.hasNext)
      var b = batch.next()
      require(!batch.hasNext)
      require(b._1.size(1) == 4)
      data(0) = b._1.valueAt(1, 1)
      data(1) = b._1.valueAt(2, 1)
      data(2) = b._1.valueAt(3, 1)
      data(3) = b._1.valueAt(4, 1)
      require(b._2.size(1) == 4)

      batch = batchIter.next()
      require(!batchIter.hasNext)
      require(batch.hasNext)
      b = batch.next()
      require(!batch.hasNext)
      require(b._1.size(1) == 4)
      data(4) = b._1.valueAt(1, 1)
      data(5) = b._1.valueAt(2, 1)
      data(6) = b._1.valueAt(3, 1)
      data(7) = b._1.valueAt(4, 1)
      require(b._2.size(1) == 4)

      val t = data.sortWith(_ < _)
      require(t(0) == 0.0)
      require(t(1) == 1.0)
      require(t(2) == 2.0)
      require(t(3) == 3.0)
      require(t(4) == 4.0)
      require(t(5) == 5.0)
      require(t(6) == 6.0)
      require(t(7) == 7.0)

      Iterator.empty
    }).count()
  }
}
