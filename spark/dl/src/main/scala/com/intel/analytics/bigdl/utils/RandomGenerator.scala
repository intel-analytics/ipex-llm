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

import java.io.{BufferedInputStream, FileInputStream}
import java.nio.ByteBuffer
import java.nio.file.{Files, Paths}

object RandomGenerator {
  val generators = new ThreadLocal[RandomGenerator]()

  // scalastyle:off methodName
  def RNG: RandomGenerator = {
    if (generators.get() == null) {
      generators.set(new RandomGenerator())
    }
    generators.get()
  }
  // scalastyle:on methodName

  def shuffle[T](data: Array[T]): Array[T] = {
    var i = 0
    val length = data.length
    while (i < length) {
      val exchange = RNG.uniform(0, length - i).toInt + i
      val tmp = data(exchange)
      data(exchange) = data(i)
      data(i) = tmp
      i += 1
    }
    data
  }
}

/**
 * A mersenne twister based fake random number generator.
 * Please refer https://en.wikipedia.org/wiki/Mersenne_Twister.
 * Note that it has its own state so it is not thread safe.
 * So you should use RandomGenerator.RNG to get a thread local instance to use.
 * That's thread-safe.
 */
class RandomGenerator private[bigdl]() {
  private val MERSENNE_STATE_N = 624
  private val MERSENNE_STATE_M = 397
  private val MARTRX_A = 0x9908b0dfL
  private val UMASK = 0x80000000L
  /* most significant w-r bits */
  private val LMASK = 0x7fffffffL
  /* least significant r bits */
  private val randomFileOS = "/dev/urandom"

  private var state: Array[Long] = new Array[Long](MERSENNE_STATE_N)
  private var seed: Long = 0
  private var next: Int = 0
  private var left: Int = 1
  private var normalX: Double = 0
  private var normalY: Double = 0
  private var normalRho: Double = 0
  private var normalIsValid: Boolean = false

  setSeed(randomSeed())

  private[bigdl] def this(seed: Long) = {
    this()
    setSeed(seed)
  }

  override def clone(): RandomGenerator = {
    val result = new RandomGenerator()
    result.copy(this)
    result
  }

  def copy(from: RandomGenerator): this.type = {
    this.state = from.state.clone()
    this.seed = from.seed
    this.next = from.next
    this.normalX = from.normalX
    this.normalY = from.normalY
    this.normalRho = from.normalRho
    this.normalIsValid = from.normalIsValid
    this
  }

  private def randomSeed(): Long = {
    if (Files.exists(Paths.get(randomFileOS))) {
      val fis = new FileInputStream(randomFileOS)
      val bis = new BufferedInputStream(fis)
      val buffer = new Array[Byte](8)
      bis.read(buffer, 0, 8)
      val randomNumber = ByteBuffer.wrap(buffer).getLong
      bis.close()
      fis.close()
      randomNumber
    }
    else {
      System.nanoTime()
    }
  }

  @inline
  private def twist(u: Long, v: Long): Long = {
    ((((u) & UMASK) | ((v) & LMASK)) >> 1) ^ (
      if ((v & 0x00000001L) != 0) {
        MARTRX_A
      } else {
        0
      }
      )
  }

  def reset(): this.type = {
    var i = 0
    while (i < MERSENNE_STATE_N) {
      this.state(i) = 0L
      i += 1
    }

    this.seed = 0
    this.next = 0
    this.normalX = 0
    this.normalY = 0
    this.normalRho = 0
    this.normalIsValid = false
    this
  }

  def setSeed(seed: Long): this.type = {
    this.reset()
    this.seed = seed
    this.state(0) = this.seed & 0xffffffffL

    var i = 1
    while (i < MERSENNE_STATE_N) {
      this.state(i) = (1812433253L * (this.state(i - 1) ^ (this.state(i - 1) >> 30)) + i)

      /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
      /* In the previous versions, mSBs of the seed affect   */
      /* only mSBs of the array state[].                        */
      /* 2002/01/09 modified by makoto matsumoto           x  */
      this.state(i) = this.state(i) & 0xffffffffL; /* for >32 bit machines */
      i += 1
    }
    this.left = 1
    this
  }

  def getSeed(): Long = {
    this.seed
  }

  private def nextState(): this.type = {
    var j: Int = MERSENNE_STATE_N - MERSENNE_STATE_M + 1
    var k: Int = 0

    this.left = MERSENNE_STATE_N
    this.next = 0

    while (j > 1) {
      j -= 1
      this.state(k) = this.state(MERSENNE_STATE_M + k) ^ twist(this.state(k), this.state(k + 1))
      k += 1
    }

    j = MERSENNE_STATE_M
    while (j > 1) {
      j -= 1
      this.state(k) = this.state(MERSENNE_STATE_M - MERSENNE_STATE_N + k) ^ twist(this.state(k),
        this.state(k + 1))
      k += 1
    }

    this.state(k) = this.state(MERSENNE_STATE_M - MERSENNE_STATE_N + k) ^ twist(this.state(k),
      this.state(0))
    this
  }

  /**
   * Generates a random number on [0,0xffffffff]-interval
   */
  private[bigdl] def random(): Long = {
    var y: Long = 0

    this.left = this.left - 1
    if (this.left == 0) {
      this.nextState()
    }

    y = this.state(0 + this.next)
    this.next = this.next + 1

    /* Tempering */
    y ^= (y >> 11)
    y ^= (y << 7) & 0x9d2c5680L
    y ^= (y << 15) & 0xefc60000L
    y ^= (y >> 18)
    y
  }

  /**
   * Generates a random number on [0, 1)-real-interval
   */
  private def basicUniform(): Double = {
    this.random() * (1.0 / 4294967296.0)
  }

  /**
   * Generates a random number on [a, b)-real-interval uniformly
   */
  def uniform(a: Double, b: Double): Double = {
    this.basicUniform() * (b - a) + a
  }

  def normal(mean: Double, stdv: Double): Double = {
    require(stdv > 0, "standard deviation must be strictly positive")

    /* This is known as the Box-Muller method */
    if (!this.normalIsValid) {
      this.normalX = this.basicUniform()
      this.normalY = this.basicUniform()
      this.normalRho = Math.sqrt(-2 * Math.log(1.0 - this.normalY))
      this.normalIsValid = true
    } else {
      this.normalIsValid = false
    }

    if (this.normalIsValid) {
      this.normalRho * Math.cos(2 * Math.PI * this.normalX) * stdv + mean
    } else {
      this.normalRho * Math.sin(2 * Math.PI * this.normalX) * stdv + mean
    }
  }

  def exponential(lambda: Double): Double = {
    -1 / lambda * Math.log(1 - this.basicUniform())
  }

  def cauchy(median: Double, sigma: Double): Double = {
    median + sigma * Math.tan(Math.PI * (this.basicUniform() - 0.5))
  }

  def logNormal(mean: Double, stdv: Double): Double = {
    val zm = mean * mean
    val zs = stdv * stdv
    require(stdv > 0, "standard deviation must be strictly positive")
    Math.exp(normal(Math.log(zm / Math.sqrt(zs + zm)), Math.sqrt(Math.log(zs / zm + 1))))
  }

  def geometric(p: Double): Int = {
    require(p >= 0 && p <= 1, "must be >= 0 and <= 1")
    ((Math.log(1 - this.basicUniform()) / Math.log(p)) + 1).toInt
  }

  def bernoulli(p: Double): Boolean = {
    require(p >= 0 && p <= 1, "must be >= 0 and <= 1")
    this.basicUniform() <= p
  }
}
