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

package com.intel.analytics.bigdl.ppml.attestation.verifier

import com.intel.analytics.bigdl.dllib.common.zooUtils
import java.io.{BufferedOutputStream, BufferedInputStream};
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import org.apache.logging.log4j.LogManager
import org.scalatest.{FlatSpec, Matchers}
import scala.io.Source
import scala.language.postfixOps
import sys.env
import sys.process._

class SGXDCAPQuoteVerifierImplSpec extends FlatSpec with Matchers {

  val logger = LogManager.getLogger(getClass)
  var tmpDir: File = _
  val sGXDCAPQuoteVerifierImplSpec = new SGXDCAPQuoteVerifierImpl()

  // SGXDCAPQuoteVerifierImplSpec
  "SGX DCAP verify Quote " should "work" in {
    if (env.contains("SGXSDK")) {
      if (env("SGXSDK").toBoolean == true) {
        val FTP_URI = if (env.contains("FTP_URI")) {
          env("FTP_URI").toString
        }
        val quoteUrl = s"$FTP_URI/bigdl/ppml/test/sgxdcap_quote.dat"
        tmpDir = zooUtils.createTmpDir("ZooPPML").toFile()
        val tmpPath = s"${tmpDir.getAbsolutePath}/SGXDCAPQuoteVerifierImplSpec"
        val dir = new File(tmpPath).getCanonicalPath
        s"wget -nv -P $dir $quoteUrl" !;
        val quotePath = s"$dir/sgxdcap_quote.dat"
        val quoteFile = new File(quotePath)
        val in = new FileInputStream(quoteFile)
        val bufIn = new BufferedInputStream(in)
        val quote = Iterator.continually(bufIn.read()).takeWhile(_ != -1).map(_.toByte).toArray
        bufIn.close()
        in.close()
        logger.info(quote)
        val verifyQuoteResult = sGXDCAPQuoteVerifierImplSpec.verifyQuote(quote)
        verifyQuoteResult shouldNot equal(-1)
        logger.info(verifyQuoteResult)
      }
    }
  }
}
