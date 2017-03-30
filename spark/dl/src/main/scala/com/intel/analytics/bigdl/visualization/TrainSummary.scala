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

package com.intel.analytics.bigdl.visualization

import com.intel.analytics.bigdl.optim.Trigger
import com.intel.analytics.bigdl.visualization.tensorboard.{FileReader, FileWriter}

import scala.collection.mutable

/**
 * Train logger for tensorboard.
 * Use optimize.setTrainSummary to enable train logger. Then the log will be saved to
 * logDir/appName/train.
 *
 * @param logDir log dir.
 * @param appName application Name.
 */
class TrainSummary(
                    logDir: String,
                    appName: String) extends Summary(logDir, appName) {
  protected val folder = s"$logDir/$appName/train"
  protected override val writer = new FileWriter(folder)
  private val triggers: mutable.HashMap[String, Trigger] = mutable.HashMap(
    "Loss" -> Trigger.severalIteration(1),
    "Throughput" -> Trigger.severalIteration(1))

  /**
   * Read scalar values to an array of triple by tag name.
   * First element of the triple is step, second is value, third is wallClockTime.
   * @param tag tag name. Supported tag names is "LearningRate", "Loss", "Throughput"
   * @return an array of triple.
   */
  override def readScalar(tag: String): Array[(Long, Float, Double)] = {
    FileReader.readScalar(folder, tag)
  }

  /**
   * Supported tag name are LearningRate, Loss, Throughput, Parameters.
   * Parameters contains weight, bias, gradWeight, gradBias, and some running status(eg.
   * runningMean and runningVar in BatchNormalization).
   *
   * Notice: By default, we record LearningRate, Loss and Throughput each iteration, while
   * recording parameters is disabled. The reason is getting parameters from workers is a
   * heavy operation when the model is very big.
   *
   * @param tag tag name
   * @param trigger trigger
   * @return
   */
  def setSummaryTrigger(tag: String, trigger: Trigger): this.type = {
    require(tag.equals("LearningRate") || tag.equals("Loss") ||
      tag.equals("Throughput") | tag.equals("Parameters"),
      s"TrainSummary: only support LearningRate, Loss, Parameters and Throughput")
    triggers(tag) = trigger
    this
  }

  /**
   * Get a trigger by tag name.
   * @param tag
   * @return
   */
  def getSummaryTrigger(tag: String): Option[Trigger] = {
    if (triggers.contains(tag)) {
      Some(triggers(tag))
    } else {
      None
    }
  }

  private[bigdl] def getScalarTriggers(): Iterator[(String, Trigger)] = {
    triggers.filter(!_._1.equals("Parameters")).toIterator
  }
}

object TrainSummary{
  def apply(logDir: String,
            appName: String): TrainSummary = {
    new TrainSummary(logDir, appName)
  }
}
