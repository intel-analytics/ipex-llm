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

package com.intel.analytics.bigdl.ppml.attestation.service

import org.apache.logging.log4j.Logger
import org.scalatest.{FlatSpec, Matchers}

class DummyAttestationServiceSpec extends FlatSpec with Matchers {

    val dummyAttestationService = new DummyAttestationService()
    val logger: Logger = dummyAttestationService.logger

    "Get Quote " should "work" in {
        val quote = dummyAttestationService.getQuoteFromServer("")
        quote shouldNot equal ("")
        logger.info(quote)
    }

    "Attest With Server " should "work" in {
        val trueResult = dummyAttestationService.attestWithServer("a_true_quote")._1
        trueResult should equal (true)
        val falseResult = dummyAttestationService.attestWithServer("a_false_quote")._1
        falseResult should equal (false)
    }
}
