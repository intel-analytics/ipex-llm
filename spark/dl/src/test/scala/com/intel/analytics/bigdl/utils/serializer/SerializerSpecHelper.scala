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

package com.intel.analytics.bigdl.utils.serializer

import java.io.{File}
import java.lang.reflect.Modifier

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.ops.{Exp => ExpOps, Pow => PowOps, Select => SelectOps, Sum => SumOps, Tile => TileOps}
import com.intel.analytics.bigdl.nn.tf.{DecodeGif => DecodeGifOps, DecodeJpeg => DecodeJpegOps, DecodePng => DecodePngOps, DecodeRaw => DecodeRawOps}
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.utils.tf.loaders.{Pack => _}
import com.intel.analytics.bigdl.utils.{Shape => KShape}
import org.reflections.Reflections
import org.reflections.scanners.SubTypesScanner
import org.reflections.util.{ClasspathHelper, ConfigurationBuilder, FilterBuilder}
import org.scalatest.{BeforeAndAfterAll, FlatSpec, Matchers}

import scala.collection.JavaConverters._
import scala.collection.mutable


abstract class SerializerSpecHelper extends FlatSpec with Matchers with BeforeAndAfterAll{

  val postFix = "bigdl"
  val excludedClass = new mutable.HashSet[String]()
  val excludedPackage = new mutable.HashSet[String]()

  private val expected = new mutable.HashSet[String]()
  val tested = new mutable.HashSet[String]()

  private var executedCount = 0

  protected def getPackage(): String = ""

  protected def addExcludedClass(): Unit = {}

  protected def addExcludedPackage(): Unit = {}

  protected def getExpected(): mutable.Set[String] = expected

  override protected def beforeAll() = {
    addExcludedClass
    addExcludedPackage
    val filterBuilder = new FilterBuilder()
    excludedPackage.foreach(filterBuilder.excludePackage(_))
    val reflections = new Reflections(new ConfigurationBuilder()
      .filterInputsBy(filterBuilder)
      .setUrls(ClasspathHelper.forPackage(getPackage()))
      .setScanners(new SubTypesScanner()))
    val subTypes = reflections.getSubTypesOf(classOf[AbstractModule[_, _, _]])
      .asScala.filter(sub => !Modifier.isAbstract(sub.getModifiers)).
      filter(sub => !excludedClass.contains(sub.getName))
    subTypes.foreach(sub => expected.add(sub.getName))
  }

  protected def runSerializationTest(module : AbstractModule[_, _, Float],
                                   input : Activity, cls: Class[_] = null) : Unit = {
    runSerializationTestWithMultiClass(module, input,
      if (cls == null) Array(module.getClass) else Array(cls))
  }

  protected def runSerializationTestWithMultiClass(module : AbstractModule[_, _, Float],
      input : Activity, classes: Array[Class[_]]) : Unit = {
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
    classes.foreach(cls => {
      if (getExpected.contains(cls.getName)) {
        tested.add(cls.getName)
      }
    })
  }


  override protected def afterAll() = {
    println(s"total ${getExpected.size}, remaining ${getExpected.size - tested.size}")
    tested.filter(!getExpected.contains(_)).foreach(t => {
      println(s"$t do not need to be tested")
    })
    getExpected.foreach(exp => {
      require(tested.contains(exp), s" $exp not included in the test!")
    })
  }
}
