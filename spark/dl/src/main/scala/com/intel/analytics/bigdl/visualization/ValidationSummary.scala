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

import com.intel.analytics.bigdl.visualization.tensorboard.{FileReader, FileWriter}

/**
 * Validation logger for tensorboard.
 * Use optimize.setValidation to enable validation logger. Then the log will be saved to
 * logDir/appName/Validation.
 *
 * @param logDir
 * @param appName
 */
class ValidationSummary(
                         logDir: String,
                         appName: String) extends Summary(logDir, appName) {
  protected val folder = s"$logDir/$appName/validation"
  protected override val writer = new FileWriter(folder)

  /**
   * ReadScalar by tag name. Optional tag name is based on ValidationMethod, "Loss",
   * "Top1Accuracy" or "Top5Accuracy".
   * @param tag tag name.
   * @return an array of triple.
   */
  override def readScalar(tag: String): Array[(Long, Float, Double)] = {
    FileReader.readScalar(folder, tag)
  }
}

object ValidationSummary{
  def apply(logDir: String,
            appName: String): ValidationSummary = {
    new ValidationSummary(logDir, appName)
  }
}
