/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.api.keras.serializer

import java.io.File

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.serializer.{ModuleLoader, ModulePersister}
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import org.scalatest.{BeforeAndAfterAll, FlatSpec, Matchers}


class SerializerSpec extends SerializerSpecHelper {
  runTests(getExpectedTests())
}

private[zoo] abstract class ModuleSerializationTest
  extends FlatSpec with Matchers with BeforeAndAfterAll{

  val postFix = "analytics-zoo"

  def test(): Unit

  protected def createTmpDir() = {
    ZooSpecHelper.createTmpDir()
  }

  protected def runSerializationTest(
      module: AbstractModule[_, _, Float],
      input: Activity, cls: Class[_] = null) : Unit = {
    runSerializationTestWithMultiClass(module, input,
      if (cls == null) Array(module.getClass) else Array(cls))
  }

  protected def runSerializationTestWithMultiClass(
      module: AbstractModule[_, _, Float],
      input: Activity, classes: Array[Class[_]]) : Unit = {
    val name = module.getName
    val serFile = File.createTempFile(name, postFix)
    val originForward = module.evaluate().forward(input)

    ModulePersister.saveToFile[Float](serFile.getAbsolutePath, null, module.evaluate(), true)
    RNG.setSeed(1000)
    val loadedModule = ModuleLoader.loadFromFile[Float](serFile.getAbsolutePath)

    val afterLoadForward = loadedModule.forward(input)

    if (serFile.exists) {
      serFile.delete
    }

    afterLoadForward should be (originForward)
  }

}
