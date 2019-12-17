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
package com.intel.analytics.bigdl.integration.torch
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.scalatest._

import scala.util.Random

case class TestCaseIdentity(value: String) {
  def suffix: String = List(".t7", value).mkString(".")
}

class TorchSpec extends FlatSpec with BeforeAndAfter with Matchers {

  implicit var testCaseIdentity: TestCaseIdentity = _

  before {
    testCaseIdentity = null
  }

  override def withFixture(test: NoArgTest): Outcome = {
    Random.setSeed(1)
    RNG.setSeed(100)

    // the identity name is class name + test case name
    val id = List(this.getClass.getName, test.name.hashCode).mkString("_")
    testCaseIdentity = TestCaseIdentity(id)
    super.withFixture(test)
  }

  protected def suffix: String = {
    testCaseIdentity.suffix
  }

  def torchCheck(): Unit = {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
  }
}

class TorchRNNSpec extends TorchSpec {
  override def torchCheck(): Unit = {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }

    if (!TH.hasRNN) {
      cancel("Torch rnn is not installed")
    }
  }
}
