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

import java.lang.reflect.Modifier

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import org.reflections.Reflections
import org.reflections.scanners.SubTypesScanner
import org.reflections.util.{ClasspathHelper, ConfigurationBuilder, FilterBuilder}

import scala.collection.JavaConverters._
import scala.collection.mutable

class SerializerSpec extends ZooSpecHelper {
  private val excluded = Set[String](
    "com.intel.analytics.zoo.pipeline.api.autograd.LambdaTorch",
    "com.intel.analytics.zoo.pipeline.api.net.TFNet")

  private val unRegularNameMapping = Map[String, String]()

  private val suffix = "SerialTest"

  private val testClasses = new mutable.HashSet[String]()

  {
    val filterBuilder = new FilterBuilder()
    val reflections = new Reflections(new ConfigurationBuilder()
      .filterInputsBy(filterBuilder)
      .addUrls(ClasspathHelper.forPackage("com.intel.analytics.zoo"))
      .addUrls(ClasspathHelper.forPackage("com.intel.analytics.bigdl.nn"))
      .setScanners(new SubTypesScanner()))

    val subTypes = reflections.getSubTypesOf(classOf[AbstractModule[_, _, _]]).asScala
      .filter(sub => !Modifier.isAbstract(sub.getModifiers))
      .filter(sub => !excluded.contains(sub.getName))
      .filter(sub => sub.getName.contains("com.intel.analytics.zoo"))
    subTypes.foreach(sub => testClasses.add(sub.getName))
  }

  private def getTestClassName(clsName: String): String = {
    if (unRegularNameMapping.contains(clsName)) {
      unRegularNameMapping(clsName)
    } else {
      clsName + suffix
    }
  }

  testClasses.foreach(cls => {
    "Serialization test of module " + cls should "be correct" in {
      val clsWholeName = getTestClassName(cls)
      try {
        val ins = Class.forName(clsWholeName)
        val testClass = ins.getConstructors()(0).newInstance()
        require(testClass.isInstanceOf[ModuleSerializationTest], s"$clsWholeName should be a " +
          s"subclass of com.intel.analytics.zoo.pipeline.api.keras.layers.serializer." +
          s"ModuleSerializationTest")
        testClass.asInstanceOf[ModuleSerializationTest].test()
      } catch {
        case t: Throwable => throw t
      }
    }
  })

}

private[zoo] abstract class ModuleSerializationTest extends SerializerSpecHelper {
  def test(): Unit
}
