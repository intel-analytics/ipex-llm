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

package com.intel.analytics.bigdl.nn

/**
 * Initialization method to initialize bias and weight
 */
sealed trait InitializationMethod

case object Default extends InitializationMethod

/**
 * In short, it helps signals reach deep into the network.
 *
 * During the training process of deep nn:
 *        1. If the weights in a network start are too small,
 *           then the signal shrinks as it passes through
 *           each layer until it’s too tiny to be useful.
 *
 *        2. If the weights in a network start too large,
 *           then the signal grows as it passes through each
 *           layer until it’s too massive to be useful.
 *
 * Xavier initialization makes sure the weights are ‘just right’,
 * keeping the signal in a reasonable range of values through many layers.
 *
 * More details on the paper
 *  [Understanding the difficulty of training deep feedforward neural networks]
 *  (http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
 */
case object Xavier extends InitializationMethod

case object BilinearFiller extends InitializationMethod
