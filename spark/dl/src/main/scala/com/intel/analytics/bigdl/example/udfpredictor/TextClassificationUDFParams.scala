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
package com.intel.analytics.bigdl.example.udfpredictor

import com.intel.analytics.bigdl.example.utils.AbstractTextClassificationParams

/**
 * Text classification udf parameters
 */
case class TextClassificationUDFParams(
                                        override val baseDir: String = "./",
                                        override val maxSequenceLength: Int = 1000,
                                        override val maxWordsNum: Int = 20000,
                                        override val trainingSplit: Double = 0.8,
                                        override val batchSize: Int = 128,
                                        override val embeddingDim: Int = 100,
                                        override val partitionNum: Int = 4,
                                        modelPath: Option[String] = None,
                                        checkpoint: Option[String] = None,
                                        testDir: String = "./")
  extends AbstractTextClassificationParams

