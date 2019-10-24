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

package com.intel.analytics.bigdl.utils

import com.intel.analytics.bigdl.mkl.hardware.Affinity
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}

import scala.concurrent.ExecutionException

class ThreadPoolSpec extends FlatSpec with Matchers {

  "mkldnn backend" should "create omp threads and bind correctly" in {
    com.intel.analytics.bigdl.mkl.MklDnn.isLoaded
    val poolSize = 1
    val ompSize = 4

    val threadPool = new ThreadPool(poolSize)
    // backup the affinities
    val affinities = threadPool.invokeAndWait2( (0 until poolSize).map(i =>
      () => {
        Affinity.getAffinity()
      })).map(_.get()).toArray

    threadPool.setMKLThreadOfMklDnnBackend(ompSize)

    // the first core can be used maybe not the 0, it depends on the affinity settings.
    threadPool.invokeAndWait2( (0 until poolSize).map( i =>
      () => {
        Affinity.getAffinity.length should be (1)
        Affinity.getAffinity.head should be (affinities.head.head)
      }))

    // set back the affinities
    threadPool.invokeAndWait2( (0 until poolSize).map( i => () => {
      Affinity.setAffinity(affinities(i))
    }))

    threadPool.invokeAndWait2( (0 until poolSize).map( i =>
      () => {
        Affinity.getAffinity.zip(affinities.head).foreach(ai => ai._1 should be (ai._2))
      }))

  }

  "mkldnn thread affinity binding" should "not influence other threads" in {
    val poolSize = 1
    val ompSize = 4

    val threadPool = new ThreadPool(poolSize)
    // backup the affinities
    val affinities = threadPool.invokeAndWait2( (0 until poolSize).map(i =>
      () => {
        Affinity.getAffinity()
    })).map(_.get()).toArray
    threadPool.setMKLThreadOfMklDnnBackend(ompSize)

    // the thread in thread pool will be set affinity to one core, which is
    // the first core can be used.
    threadPool.invokeAndWait2( (0 until poolSize).map( i =>
      () => {
        Affinity.getAffinity.length should be (1)
        Affinity.getAffinity.head should be (affinities.head.head)
      }))

    val threadPool2 = new ThreadPool(poolSize)
    // the thread has not been set any affinities, so it should return all
    // cores can be used.
    threadPool2.invokeAndWait2( (0 until poolSize).map(i => () => {
      println(Affinity.getAffinity.mkString("\t"))
      Affinity.getAffinity.length should be (affinities.head.length)
    }))
  }

  "invokeAndWait2" should "catch the unsupported exception" in {
    val threadPool = new ThreadPool(1)
    val task = () => { throw new UnsupportedOperationException(s"test invokeAndWait2") }

    intercept[UnsupportedOperationException] {
      threadPool.invokeAndWait2( (0 until 1).map( i => task ))
    }
  }

  "invokeAndWait2" should "catch the interrupt exception" in {
    val threadPool = new ThreadPool(1)
    val task = () => { throw new InterruptedException(s"test invokeAndWait2")}

    intercept[InterruptedException] {
      threadPool.invokeAndWait2( (0 until 1).map( i => task ))
    }
  }

  "invokeAndWait" should "catch the exception" in {
    val threadPool = new ThreadPool(1)
    val task = () => { throw new InterruptedException(s"test invokeAndWait")}

    intercept[InterruptedException] {
      threadPool.invokeAndWait( (0 until 1).map( i => task ))
    }
  }

  "invoke" should "catch the exception" in {
    val threadPool = new ThreadPool(1)
    val task = () => { throw new UnsupportedOperationException(s"test invoke2") }

    intercept[ExecutionException] {
      val results = threadPool.invoke2( (0 until 1).map( i => task ))
      results.foreach(_.get())
    }
  }

  "setFlushDenormalState" should "influence float arithmetic operations" in {
    val poolSize = 1
    val ompSize = 4

    val threadPool = new ThreadPool(poolSize)

    System.setProperty("bigdl.flushDenormalState", "false")
    threadPool.setMKLThreadOfMklDnnBackend(ompSize)

    threadPool.invokeAndWait2( (0 until 1).map(i => () => {
      // A denormalized value is in the range
      // from 1.4E-45 to 1.18E-38,
      // or from -1.18E-38 to -1.4E-45
      val denormal: Float = -1.234E-41F
      val floatOne: Float = 1.0F

      val result: Float = denormal * floatOne
      // The result should not be zero without setting
      (result == 0.0F) should be(false)
    }))

    System.setProperty("bigdl.flushDenormalState", "true")
    threadPool.setMKLThreadOfMklDnnBackend(ompSize)

    threadPool.invokeAndWait2( (0 until 1).map(i => () => {
      // A denormalized value is in the range
      // from 1.4E-45 to 1.18E-38,
      // or from -1.18E-38 to -1.4E-45
      val denormal: Float = -1.234E-41F
      val floatOne: Float = 1.0F

      val result = denormal * floatOne
      // The result should be zero with setting
      (result == 0.0F) should be (true)
    }))

    System.clearProperty("bigdl.flushDenormalState")
  }

  "setFlushDenormalState" should "not influence other threads float arithmetic operations" in {
    val poolSize = 1
    val ompSize = 4

    val threadPool1 = new ThreadPool(poolSize)
    System.setProperty("bigdl.flushDenormalState", "true")
    threadPool1.setMKLThreadOfMklDnnBackend(ompSize)

    val threadPool2 = new ThreadPool(poolSize)
    System.setProperty("bigdl.flushDenormalState", "false")
    threadPool2.setMKLThreadOfMklDnnBackend(ompSize)

    // A denormalized value is in the range
    // from 1.4E-45 to 1.18E-38,
    // or from -1.18E-38 to -1.4E-45
    val denormal: Float = -1.234E-41F
    val floatOne: Float = 1.0F

    threadPool1.invokeAndWait2( (0 until 1).map(i => () => {
      val result = denormal * floatOne
      // The result should be zero with setting
      (result == 0.0F) should be (true)
    }))

    threadPool2.invokeAndWait2( (0 until 1).map(i => () => {
      val result = denormal * floatOne
      // The result should not be zero without setting
      (result == 0.0F) should be (false)
    }))

    System.clearProperty("bigdl.flushDenormalState")
  }

  "setFlushDenormalState" should "not influence other threads MKL operations" in {
    val poolSize = 1
    val ompSize = 4

    val threadPool1 = new ThreadPool(poolSize)
    System.setProperty("bigdl.flushDenormalState", "false")
    threadPool1.setMKLThreadOfMklDnnBackend(ompSize)

    val threadPool2 = new ThreadPool(poolSize)
    System.setProperty("bigdl.flushDenormalState", "true")
    threadPool2.setMKLThreadOfMklDnnBackend(ompSize)

    val tensor = Tensor[Float](1).fill(-1.234E-41F)
    val floatOne = 1.0F

    threadPool1.invokeAndWait2( (0 until 1).map(i => () => {
      val result = tensor.pow(floatOne)

      for (i <- 0 to result.storage().array().length -1) {
        assert(result.storage().array()(i) != 0.0F)
      }
    }))

    System.clearProperty("bigdl.flushDenormalState")
  }
}
