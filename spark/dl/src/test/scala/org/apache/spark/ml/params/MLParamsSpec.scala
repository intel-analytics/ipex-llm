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
package org.apache.spark.ml.params

import com.intel.analytics.bigdl.optim._
import org.apache.spark.ml.param.{OptimMethodParam, ParamMap, Params, TriggerParam}
import org.scalatest.{FlatSpec, Matchers}

class MLParamsSpec extends FlatSpec with Matchers {

  private class TestMLParams extends Params {

    final val opt: OptimMethodParam =
      new OptimMethodParam(this, "optimMethod", "Hello gays! I'm a cool optimMethod!")

    def setOpt(optim: OptimMethod[_]): Unit = set(opt -> optim)

    def getOpt: OptimMethod[_] = $(opt)

    final val trigger: TriggerParam =
      new TriggerParam(this, "trigger", "Hello gays! I'm a cute trigger!")

    def setTrigger(_trigger: Trigger): Unit = set(trigger -> _trigger)

    def getTrigger: Trigger = $(trigger)

    override def copy(extra: ParamMap): Params = copyValues(new TestMLParams)

    override val uid: String = "TestMLParams"

  }

  private val params = new TestMLParams

  import scala.reflect.runtime.{universe => ru}
  private val mirror = ru.runtimeMirror(params.getClass.getClassLoader)
  private val getType = (any: Any) => mirror.classSymbol(any.getClass).toType

  "AdadeltaBuilder" should "do encoding/decoding correctly" in {
    val opt = new Adadelta[Double](.3, 1e-8)
    val jsonStr = params.opt.jsonEncode(opt)
    val optDecoded = params.opt.jsonDecode(jsonStr).asInstanceOf[Adadelta[Double]]
    getType(opt) shouldEqual getType(optDecoded)
    opt.Epsilon shouldEqual optDecoded.Epsilon
    opt.decayRate shouldEqual optDecoded.decayRate
  }

  "AdagradBuilder" should "do encoding/decoding correctly" in {
    val opt = new Adagrad[Double](1e-2, .1, .01)
    val jsonStr = params.opt.jsonEncode(opt)
    val optDecoded = params.opt.jsonDecode(jsonStr).asInstanceOf[Adagrad[Double]]
    getType(opt) shouldEqual getType(optDecoded)
    opt.learningRate shouldEqual optDecoded.learningRate
    opt.learningRateDecay shouldEqual optDecoded.learningRateDecay
    opt.weightDecay shouldEqual optDecoded.weightDecay
  }

  "AdamBuilder" should "do encoding/decoding correctly" in {
    val opt = new Adam[Double](1e-2, .1, .91, .99, 1e-7)
    val jsonStr = params.opt.jsonEncode(opt)
    val optDecoded = params.opt.jsonDecode(jsonStr).asInstanceOf[Adam[Double]]
    getType(opt) shouldEqual getType(optDecoded)
    opt.learningRate shouldEqual optDecoded.learningRate
    opt.learningRateDecay shouldEqual optDecoded.learningRateDecay
    opt.beta1 shouldEqual optDecoded.beta1
    opt.beta2 shouldEqual optDecoded.beta2
    opt.Epsilon shouldEqual optDecoded.Epsilon
  }

  "AdamaxBuilder" should "do encoding/decoding correctly" in {
    val opt = new Adamax[Float](.001, .92, .99, 1e-37)
    val jsonStr = params.opt.jsonEncode(opt)
    val optDecoded = params.opt.jsonDecode(jsonStr).asInstanceOf[Adamax[Float]]
    getType(opt) shouldEqual getType(optDecoded)
    opt.learningRate shouldEqual optDecoded.learningRate
    opt.beta1 shouldEqual optDecoded.beta1
    opt.beta2 shouldEqual optDecoded.beta2
    opt.Epsilon shouldEqual optDecoded.Epsilon
  }

  "RMSpropBuilder" should "do encoding/decoding correctly" in {
    val opt = new RMSprop[Double](.001, 1e-3, .999, 1e-9)
    val jsonStr = params.opt.jsonEncode(opt)
    val optDecoded = params.opt.jsonDecode(jsonStr).asInstanceOf[RMSprop[Double]]
    getType(opt) shouldEqual getType(optDecoded)
    opt.learningRate shouldEqual optDecoded.learningRate
    opt.learningRateDecay shouldEqual optDecoded.learningRateDecay
    opt.decayRate shouldEqual optDecoded.decayRate
    opt.Epsilon shouldEqual optDecoded.Epsilon
  }

  "LBFGSBuilder" should "do encoding/decoding correctly" in {
    val opt = new LBFGS[Float](10, 1e10, 1e-4, 1e-8, 90, .99, true)
    val jsonStr = params.opt.jsonEncode(opt)
    val optDecoded = params.opt.jsonDecode(jsonStr).asInstanceOf[LBFGS[Float]]
    getType(opt) shouldEqual getType(optDecoded)
    opt.maxIter shouldEqual optDecoded.maxIter
    opt.maxEval shouldEqual optDecoded.maxEval
    opt.tolFun shouldEqual optDecoded.tolFun
    opt.tolX shouldEqual optDecoded.tolX
    opt.nCorrection shouldEqual optDecoded.nCorrection
    opt.learningRate shouldEqual optDecoded.learningRate
    opt.verbose shouldEqual optDecoded.verbose
    opt.lineSearch shouldEqual optDecoded.lineSearch
    opt.lineSearchOptions shouldEqual optDecoded.lineSearchOptions
  }

  "SGDBuilder" should "do encoding/decoding correctly" in {
    val opt = new SGD[Double](.001, 1e-3, .999, 1e-9)
    val jsonStr = params.opt.jsonEncode(opt)
    val optDecoded = params.opt.jsonDecode(jsonStr).asInstanceOf[SGD[Double]]
    getType(opt) shouldEqual getType(optDecoded)
    val opt2 = new SGD[Float](.001, 1e-3, .999, 1e-9)
    getType(opt2) should not be getType(optDecoded)
    opt.learningRate shouldEqual optDecoded.learningRate
    opt.learningRateDecay shouldEqual optDecoded.learningRateDecay
    opt.weightDecay shouldEqual optDecoded.weightDecay
    opt.momentum shouldEqual optDecoded.momentum
    opt.dampening shouldEqual optDecoded.dampening
    opt.nesterov shouldEqual optDecoded.nesterov
    opt.learningRateSchedule shouldEqual optDecoded.learningRateSchedule
    opt.learningRates shouldEqual optDecoded.learningRates
    opt.weightDecays shouldEqual optDecoded.weightDecays
  }

  "TriggerParam" should "do encoding/decoding correctly" in {
    val everyEpoch = Trigger.everyEpoch
    var str = params.trigger.jsonEncode(everyEpoch)
    getType(everyEpoch) shouldEqual getType(params.trigger.jsonDecode(str))

    val severalIteration = Trigger.severalIteration(100)
    str = params.trigger.jsonEncode(severalIteration)
    getType(severalIteration) shouldEqual getType(params.trigger.jsonDecode(str))

    val maxEpoch = Trigger.maxEpoch(100)
    str = params.trigger.jsonEncode(maxEpoch)
    getType(maxEpoch) shouldEqual getType(params.trigger.jsonDecode(str))

    val maxIteration = Trigger.maxIteration(100)
    str = params.trigger.jsonEncode(maxIteration)
    getType(maxIteration) shouldEqual getType(params.trigger.jsonDecode(str))

    val maxScore = Trigger.maxScore(100.0f)
    str = params.trigger.jsonEncode(maxScore)
    getType(maxScore) shouldEqual getType(params.trigger.jsonDecode(str))

    val minLoss = Trigger.minLoss(100.0f)
    str = params.trigger.jsonEncode(minLoss)
    getType(minLoss) shouldEqual getType(params.trigger.jsonDecode(str))
  }
}
