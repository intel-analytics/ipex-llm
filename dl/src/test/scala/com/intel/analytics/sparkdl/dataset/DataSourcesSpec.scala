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
import java.nio.file.Paths

import org.scalatest.{FlatSpec, Matchers}

class DataSourcesSpec extends FlatSpec with Matchers {
  private def processPath(path: String): String = {
    if (path.contains(":")) {
      path.substring(1)
    } else {
      path
    }
  }

  "mnist data source" should "load image correct" in {
    val resource = getClass().getClassLoader().getResource("mnist")

    val dataSource = new MNISTDataSource(
      processPath(resource.getPath()) + File.separator + "t10k-images.idx3-ubyte",
      processPath(resource.getPath()) + File.separator + "t10k-labels.idx1-ubyte",
      looped = false
    )
    dataSource.total() should be(10000)
    dataSource.map(_.label()).min should be(1.0f)
    dataSource.reset()
    dataSource.map(_.label()).max should be(10.0f)
  }

  "cifar data source" should "load image correct" in {
    val resource = getClass().getClassLoader().getResource("cifar")
    val dataSource = new CifarDataSource(Paths.get(processPath(resource.getPath())), looped = false)
    dataSource.total() should be(7)
    val labelMap = dataSource.getLabelMap(Paths.get(processPath(resource.getPath())))
    labelMap("airplane") should be(1)
    labelMap("deer") should be(2)

    val img1 = dataSource.next()
    img1.label() should be(1f)
    img1.content(2) should be(234 / 255f)
    img1.content(1) should be(125 / 255f)
    img1.content(0) should be(59 / 255f)
    img1.content((22 + 4 * 32) * 3 + 2) should be(253 / 255f)
    img1.content((22 + 4 * 32) * 3 + 1) should be(148 / 255f)
    img1.content((22 + 4 * 32) * 3) should be(31 / 255f)
    val img2 = dataSource.next()
    img2.label() should be(1f)
    val img3 = dataSource.next()
    img3.label() should be(2f)
    val img4 = dataSource.next()
    img4.label() should be(2f)
    img4.content((9 + 8 * 32) * 3 + 2) should be(40 / 255f)
    img4.content((9 + 8 * 32) * 3 + 1) should be(51 / 255f)
    img4.content((9 + 8 * 32) * 3) should be(37 / 255f)
    val img5 = dataSource.next()
    img5.label() should be(2f)
    val img6 = dataSource.next()
    img6.label() should be(2f)
    val img7 = dataSource.next()
    img7.label() should be(1f)
  }

  "imagenet data source" should "load image correct" in {
    val resource = getClass().getClassLoader().getResource("imagenet")
    val dataSource = new ImageNetDataSource(Paths.get(processPath(resource.getPath())), looped =
      false)
    dataSource.total() should be(8)

    val labelMap = dataSource.getLabelMap(Paths.get(processPath(resource.getPath())))
    labelMap("n02110063") should be(1)
    labelMap("n04370456") should be(2)
    labelMap("n15075141") should be(3)

    val img1 = dataSource.next()
    img1.label() should be(1f)
    (img1.width() == 256 || img1.height() == 256) should be(true)
    val path1 = java.io.File.createTempFile("UnitTest", "datasource1.jpg").getAbsolutePath
    img1.save(path1)
    println(s"save test image to $path1")

    val img2 = dataSource.next()
    img2.label() should be(1f)
    (img2.width() == 256 || img2.height() == 256) should be(true)
    val path2 = java.io.File.createTempFile("UnitTest", "datasource2.jpg").getAbsolutePath
    img1.save(path2)
    println(s"save test image to $path2")

    val img3 = dataSource.next()
    img3.label() should be(1f)
    (img3.width() == 256 || img3.height() == 256) should be(true)

    val img4 = dataSource.next()
    img4.label() should be(2f)
    (img4.width() == 256 || img4.height() == 256) should be(true)

    val img5 = dataSource.next()
    img5.label() should be(2f)
    (img5.width() == 256 || img5.height() == 256) should be(true)

    val img6 = dataSource.next()
    img6.label() should be(3f)
    (img6.width() == 256 || img6.height() == 256) should be(true)

    val img7 = dataSource.next()
    img7.label() should be(3f)
    (img7.width() == 256 || img7.height() == 256) should be(true)

    val img8 = dataSource.next()
    img8.label() should be(3f)
    (img8.width() == 256 || img8.height() == 256) should be(true)
  }
}
