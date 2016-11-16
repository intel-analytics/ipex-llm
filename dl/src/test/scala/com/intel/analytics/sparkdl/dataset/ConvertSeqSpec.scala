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
package com.intel.analytics.sparkdl.dataset

import java.io.File
import java.net.URI
import java.nio.file.Paths

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.{SequenceFile, Text, Writable}
import org.apache.hadoop.util.ReflectionUtils
import org.scalatest.{FlatSpec, Matchers}

class ConvertSeqSpec extends FlatSpec with Matchers {
  import Utils._

  "convert ImageNet Image " should "correct" in {
    val parallel = 1
    val tmpFile = java.io.File.createTempFile("seq", "tmp")
    val output = tmpFile.toString
    val resource = getClass().getClassLoader().getResource("imagenet")
    val dataSource =
      new ImageNetDataSource(Paths.get(processPath(resource.getPath())), looped = false)
    val pathToImage = PathToRGBImage(256)
    val worker = new Worker(dataSource -> pathToImage, parallel)
    worker.process(output)

    dataSource.reset()
    val uri = s"${output}-seq"
    val path = new Path(uri)
    val conf = new Configuration
    val fs = FileSystem.get(new File(uri).toURI, conf)
    val reader = new SequenceFile.Reader(fs, path, conf)
    val key = ReflectionUtils.newInstance(reader.getKeyClass, conf).asInstanceOf[Writable]
    val value = new Text
    var position = reader.getPosition
    while (reader.next(key, value)) {
      val data = value.getBytes
      val tmpImage = (dataSource -> pathToImage).next()
      val dataImage = tmpImage.content
      data(1000 + 8) should be((dataImage(1000) * 255).toByte)
      data(5000 + 8) should be((dataImage(5000) * 255).toByte)
      data(10000 + 8) should be((dataImage(10000) * 255).toByte)
      data(15000 + 8) should be((dataImage(15000) * 255).toByte)
      data(20000 + 8) should be((dataImage(20000) * 255).toByte)
      position = reader.getPosition
    }
  }

  "convert Cifar Image " should "correct" in {
    val parallel = 1
    val tmpFile = java.io.File.createTempFile("seq", "tmp")
    val output = tmpFile.toString
    val resource = getClass().getClassLoader().getResource("cifar")
    val dataSource =
      new CifarDataSource(Paths.get(processPath(resource.getPath())), looped = false)
    val arrayToImage = ArrayByteToRGBImage()
    val worker = new Worker(dataSource -> arrayToImage, parallel)
    worker.process(output)

    dataSource.reset()
    val uri = s"${output}-seq"
    val path = new Path(uri)
    val conf = new Configuration
    val fs = FileSystem.get(new File(uri).toURI, conf)
    val reader = new SequenceFile.Reader(fs, path, conf)
    val key = ReflectionUtils.newInstance(reader.getKeyClass, conf).asInstanceOf[Writable]
    val value = new Text
    var position = reader.getPosition
    while (reader.next(key, value)) {
      val data = value.getBytes
      val tmpImage = (dataSource -> arrayToImage).next()
      val dataImage = tmpImage.content
      data(100 + 8) should be((dataImage(100) * 255.0f).toByte)
      data(500 + 8) should be((dataImage(500) * 255.0f).toByte)
      data(1000 + 8) should be((dataImage(1000) * 255.0f).toByte)
      data(1500 + 8) should be((dataImage(1500) * 255.0f).toByte)
      data(2000 + 8) should be((dataImage(2000) * 255.0f).toByte)
      data(2500 + 8) should be((dataImage(2500) * 255.0f).toByte)
      position = reader.getPosition
    }
  }
}
