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

package com.intel.analytics.bigdl.dataset

import com.intel.analytics.bigdl.dataset.text.{BLEU, SmoothingFunction}
import org.scalatest.{FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Parallel
class BLEUSpec extends FlatSpec with Matchers {
  "BLEUSpec" should " calculate correctly" in {
    val hypothesis1 = Array("It", "is", "a", "guide", "to", "action", "which",
      "ensures", "that", "the", "military", "always",
      "obeys", "the", "commands", "of", "the", "party")
    val hypothesis2 = Array("It", "is", "to", "insure", "the", "troops", "forever",
      "hearing", "the", "activity", "guidebook", "that", "party", "direct")

    val reference1 = Array("It", "is", "a", "guide", "to", "action", "that",
      "ensures", "that", "the", "military", "will",
      "forever", "heed", "party", "commands")

    val reference2 = Array("It", "is", "the", "guiding", "principle", "which", "guarantees",
      "the", "military", "forces", "always", "being",
      "under", "the", "command", "of", "the", "party")

    val reference3 = Array("It", "is", "the", "practical", "guide", "for", "the",
      "army", "always", "to", "heed", "the",
      "directions", "of", "the", "party")

    val bleu = BLEU()
    val score1 = bleu.sentenceBleu(Array(reference1, reference2, reference3), hypothesis1)
    val score2 = bleu.sentenceBleu(Array(reference1, reference2, reference3), hypothesis2)

    score1 should be ((0.5045666840058485) +- 1e-5)
    score2 should be ((0.39692877231857493) +- 1e-5)

    val score3 = bleu.sentenceBleu(Array(reference1, reference2, reference3), hypothesis1,
      weights = Array(0.1666, 0.1666, 0.1666, 0.1666, 0.1666))
    score3 should be ((0.45838627164939455) +- 1e-5)
  }

  "BLEUSpec" should " calculate correctly in corpusBleu" in {
    val hyp1 = Array("It", "is", "a", "guide", "to", "action", "which",
      "ensures", "that", "the", "military", "always",
      "obeys", "the", "commands", "of", "the", "party")

    val ref1a = Array("It", "is", "a", "guide", "to", "action", "that",
      "ensures", "that", "the", "military", "will",
      "forever", "heed", "party", "commands")

    val ref1b = Array("It", "is", "the", "guiding", "principle", "which", "guarantees",
      "the", "military", "forces", "always", "being",
      "under", "the", "command", "of", "the", "party")

    val ref1c = Array("It", "is", "the", "practical", "guide", "for", "the",
      "army", "always", "to", "heed", "the",
      "directions", "of", "the", "party")


    val hyp2 = Array("he", "read", "the", "book", "because", "he", "was",
      "interested", "in", "world", "history")

    val ref2a = Array("he", "was", "interested", "in", "world", "history",
      "because", "he", "read", "the", "book")

    val bleu = BLEU()
    val listOfRefs = Array(Array(ref1a, ref1b, ref1c), Array(ref2a))
    val hypotheses = Array(hyp1, hyp2)
    val score1 = bleu.corpusBleu(listOfRefs, hypotheses)

    val score2 = bleu.sentenceBleu(Array(ref1a, ref1b, ref1c), hyp1)
    val score3 = bleu.sentenceBleu(Array(ref2a), hyp2)
    val score4 = (score2 + score3) / 2

    score1 should be ((0.5920778868801042) +- 1e-5)
    score4 should be ((0.6223247442490669) +- 1e-5)
  }

  "BLEUSpec" should " calculate correctly in BrevityPenalty 1" in {
    val reference1 = Array.fill[String](12)("a")
    val reference2 = Array.fill[String](15)("a")
    val reference3 = Array.fill[String](17)("a")
    val hypothesis = Array.fill[String](12)("a")


    val bleu = BLEU()
    val hypLen = hypothesis.length
    val closestRefLen = bleu.closestRefLength(Array(reference1, reference2, reference3), hypLen)
    val bp = bleu.brevityPenalty(closestRefLen, hypLen)

    bp should be(1.0)
  }

  "BLEUSpec" should " calculate correctly in BrevityPenalty 2" in {
    val reference1 = Array.fill[String](28)("a")
    val reference2 = Array.fill[String](28)("a")
    val hypothesis = Array.fill[String](12)("a")


    val bleu = BLEU()
    val hypLen = hypothesis.length
    val closestRefLen = bleu.closestRefLength(Array(reference1, reference2), hypLen)
    val bp = bleu.brevityPenalty(closestRefLen, hypLen)

    bp should be ((0.2635971381157267) +- 1e-5)
  }

  "BLEUSpec" should " calculate correctly in BrevityPenalty 3" in {
    val reference1 = Array.fill[String](13)("a")
    val reference2 = Array.fill[String](2)("a")
    val hypothesis = Array.fill[String](12)("a")


    val bleu = BLEU()
    val hypLen = hypothesis.length
    val closestRefLen = bleu.closestRefLength(Array(reference1, reference2), hypLen)
    val bp = bleu.brevityPenalty(closestRefLen, hypLen)

    bp should be ((0.9200444146293233) +- 1e-5)
  }

  "BLEUSpec" should " calculate correctly in method01234567" in {
    val hyp1 = Array("It", "is", "a", "guide", "to", "action", "which",
      "ensures", "that", "the", "military", "always",
      "obeys", "the", "commands", "of", "the", "party")

    val ref1a = Array("It", "is", "a", "guide", "to", "action", "that",
      "ensures", "that", "the", "military", "will",
      "forever", "heed", "Party", "commands")

    val bleu = BLEU()
    val chencherry = SmoothingFunction()
    val scorex = bleu.sentenceBleu(Array(ref1a), hyp1)
    val score0 = bleu.sentenceBleu(Array(ref1a), hyp1, smoothingFunction = chencherry.method0())
    val score1 = bleu.sentenceBleu(Array(ref1a), hyp1, smoothingFunction = chencherry.method1())
    val score2 = bleu.sentenceBleu(Array(ref1a), hyp1, smoothingFunction = chencherry.method2())
    val score3 = bleu.sentenceBleu(Array(ref1a), hyp1, smoothingFunction = chencherry.method3())
    val score4 = bleu.sentenceBleu(Array(ref1a), hyp1, smoothingFunction = chencherry.method4())
    val score5 = bleu.sentenceBleu(Array(ref1a), hyp1, smoothingFunction = chencherry.method5())
    val score6 = bleu.sentenceBleu(Array(ref1a), hyp1, smoothingFunction = chencherry.method6())
    val score7 = bleu.sentenceBleu(Array(ref1a), hyp1, smoothingFunction = chencherry.method7())

    scorex should be ((0.41180376356915777) +- 1e-5)
    score0 should be ((0.41180376356915777) +- 1e-5)
    score1 should be ((0.41180376356915777) +- 1e-5)
    score2 should be ((0.44897710722021167) +- 1e-5)
    score3 should be ((0.41180376356915777) +- 1e-5)
    score4 should be ((0.41180376356915777) +- 1e-5)
    score5 should be ((0.49053283797997127) +- 1e-5)
    score6 should be ((0.4135895888868294) +- 1e-5)
    score7 should be ((0.49053283797997127) +- 1e-5)


  }
}
