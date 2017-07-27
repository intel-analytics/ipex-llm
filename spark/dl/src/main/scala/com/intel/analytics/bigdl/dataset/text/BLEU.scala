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
package com.intel.analytics.bigdl.dataset.text

import org.apache.commons.lang.math.Fraction

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

class BLEU() extends Serializable {

  /**
   * Calculate BLEU score (Bilingual Evaluation Understudy) from
   * Papineni, Kishore, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002.
   * "BLEU: a method for automatic evaluation of machine translation."
   * In Proceedings of ACL. http://www.aclweb.org/anthology/P02-1040.pdf
   *
   *
   * hypothesis1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
   * ...               'ensures', 'that', 'the', 'military', 'always',
   * ...               'obeys', 'the', 'commands', 'of', 'the', 'party']
   *
   *  hypothesis2 = ['It', 'is', 'to', 'insure', 'the', 'troops',
   * ...               'forever', 'hearing', 'the', 'activity', 'guidebook',
   * ...               'that', 'party', 'direct']
   *
   * reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
   * ...               'ensures', 'that', 'the', 'military', 'will', 'forever',
   * ...               'heed', 'Party', 'commands']
   *
   * reference2 = ['It', 'is', 'the', 'guiding', 'principle', 'which',
   * ...               'guarantees', 'the', 'military', 'forces', 'always',
   * ...               'being', 'under', 'the', 'command', 'of', 'the',
   * ...               'Party']
   *
   * reference3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
   * ...               'army', 'always', 'to', 'heed', 'the', 'directions',
   * ...               'of', 'the', 'party']
   *
   * sentenceBleu([reference1, reference2, reference3], hypothesis1)
   * 0.5045...
   *
   * sentenceBleu([reference1, reference2, reference3], hypothesis2)
   * 0.3969...
   *
   *
   *  The default BLEU calculates a score for up to 4grams using uniform
   * weights. To evaluate your translations with higher/lower order ngrams,
   * use customized weights. E.g. when accounting for up to 6grams with uniform
   * weights:
   *
   *  weights = Array(0.1666, 0.1666, 0.1666, 0.1666, 0.1666)
   *
   * @param references reference sentences
   * @param hypothesis a hypothesis sentence
   * @param weights
   * @param smoothingFunction
   * @param autoReweight
   * @param emulateMultibleu
   * @return
   */
  def sentenceBleu(
    references: Array[Array[String]],
    hypothesis: Array[String],
    weights: Array[Double] = Array(0.25, 0.25, 0.25, 0.25),
    smoothingFunction: (Array[Fraction]) => Array[Fraction] = null,
    autoReweight: Boolean = false,
    emulateMultibleu: Boolean = false): Double = {
      corpusBleu(
        Array(references),
        Array(hypothesis),
        weights,
        smoothingFunction,
        autoReweight,
        emulateMultibleu)
  }

  /**
   * Calculate a single corpus-level BLEU score (aka. system-level BLEU) for all
   * the hypotheses and their respective references.
   *
   * Instead of averaging the sentence level BLEU scores (i.e. marco-average
   * precision), the original BLEU metric (Papineni et al. 2002) accounts for
   * the micro-average precision (i.e. summing the numerators and denominators
   * for each hypothesis-reference(s) pairs before the division).
   *
   * hyp1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
   * ...         'ensures', 'that', 'the', 'military', 'always',
   * ...         'obeys', 'the', 'commands', 'of', 'the', 'party']
   * ref1a = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
   * ...          'ensures', 'that', 'the', 'military', 'will', 'forever',
   * ...          'heed', 'Party', 'commands']
   * ref1b = ['It', 'is', 'the', 'guiding', 'principle', 'which',
   * ...          'guarantees', 'the', 'military', 'forces', 'always',
   * ...          'being', 'under', 'the', 'command', 'of', 'the', 'Party']
   * ref1c = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
   * ...          'army', 'always', 'to', 'heed', 'the', 'directions',
   * ...          'of', 'the', 'party']
   *
   * hyp2 = ['he', 'read', 'the', 'book', 'because', 'he', 'was',
   * ...         'interested', 'in', 'world', 'history']
   * ref2a = ['he', 'was', 'interested', 'in', 'world', 'history',
   * ...          'because', 'he', 'read', 'the', 'book']
   *
   * list_of_references = [[ref1a, ref1b, ref1c], [ref2a]]
   * hypotheses = [hyp1, hyp2]
   * corpus_bleu(list_of_references, hypotheses) # doctest: +ELLIPSIS
   * 0.5920...
   *
   * The example below show that corpus_bleu() is different from averaging
   * sentence_bleu() for hypotheses
   *
   * score1 = sentence_bleu([ref1a, ref1b, ref1c], hyp1)
   * score2 = sentence_bleu([ref2a], hyp2)
   * (score1 + score2) / 2 # doctest: +ELLIPSIS
   * 0.6223...
   *
   * @param references
   * @param hypotheses
   * @param weights
   * @param smoothingFunction
   * @param autoReweight
   * @param emulateMultibleu
   * @return
   */
  def corpusBleu(
    references: Array[Array[Array[String]]],
    hypotheses: Array[Array[String]],
    weights: Array[Double] = Array(0.25, 0.25, 0.25, 0.25),
    smoothingFunction: (Array[Fraction]) => Array[Fraction] = null,
    autoReweight: Boolean = false,
    emulateMultibleu: Boolean = false): Double = {

    var _weights = weights
    val pNumerators = new Array[Int](_weights.length)
    val pDenominators = new Array[Int](_weights.length)
    var hypLengths = 0
    var refLengths = 0

    require(references.length == hypotheses.length,
      "The number of hypotheses and their reference(s) should be the same, " +
      s"# of hypotheses = ${references.length}, # of references = ${hypotheses.length}")

    references.zip(hypotheses).foreach(x => {
      val reference = x._1
      val hypothesis = x._2
      var i = 0
      while (i < _weights.length) {
        val pI = modifiedPrecision(reference, hypothesis, i + 1)
        pNumerators(i) += pI.getNumerator
        pDenominators(i) += pI.getDenominator
        i += 1
      }
      val hypLen = hypothesis.length
      hypLengths += hypLen
      refLengths += closestRefLength(reference, hypLen)
    })

    val bp = brevityPenalty(refLengths, hypLengths)

    if (autoReweight) {
      if (hypLengths < 4 && _weights == Array(0.25, 0.25, 0.25, 0.25)) {
        _weights = Array.fill[Double](hypLengths)(1 / hypLengths)
      }
    }

    val pN = pNumerators zip pDenominators map(x => Fraction.getFraction(x._1, x._2))

    if (pNumerators(0) == 0.0) {
      return 0.0
    }

    var _smoothingFunction = smoothingFunction
    if (null == _smoothingFunction) {
      _smoothingFunction = SmoothingFunction().method0()
    }

    val pNsmooth = _smoothingFunction(pN)
    val s = _weights.zip(pNsmooth)
      .map(x => x._1 * math.log(x._2.doubleValue))
      .foldLeft(0.0)((a, b) => a + b)
    val s_sum = bp * math.exp(s)

    s_sum
  }

  /**
   * Calculate brevity penalty.
   *
   * As the modified n-gram precision still has the problem from the short
   * length sentence, brevity penalty is used to modify the overall BLEU
   * score according to length.
   *
   * An example from the paper. There are three references with length 12, 15
   * and 17. And a concise hypothesis of the length 12. The brevity penalty is 1.
   *
   *     >>> reference1 = list('aaaaaaaaaaaa')      # i.e. ['a'] * 12
   *     >>> reference2 = list('aaaaaaaaaaaaaaa')   # i.e. ['a'] * 15
   *     >>> reference3 = list('aaaaaaaaaaaaaaaaa') # i.e. ['a'] * 17
   *     >>> hypothesis = list('aaaaaaaaaaaa')      # i.e. ['a'] * 12
   *    >>> references = [reference1, reference2, reference3]
   *     >>> hyp_len = len(hypothesis)
   *     >>> closest_ref_len =  closest_ref_length(references, hyp_len)
   *     >>> brevity_penalty(closest_ref_len, hyp_len)
   *     1.0
   *
   * In case a hypothesis translation is shorter than the references, penalty is
   * applied.
   *
   *     >>> references = [['a'] * 28, ['a'] * 28]
   *     >>> hypothesis = ['a'] * 12
   *     >>> hyp_len = len(hypothesis)
   *     >>> closest_ref_len =  closest_ref_length(references, hyp_len)
   *     >>> brevity_penalty(closest_ref_len, hyp_len)
   *     0.2635971381157267
   *
   * The length of the closest reference is used to compute the penalty. If the
   * length of a hypothesis is 12, and the reference lengths are 13 and 2, the
   * penalty is applied because the hypothesis length (12) is less then the
   * closest reference length (13).
   *
   *     >>> references = [['a'] * 13, ['a'] * 2]
   *     >>> hypothesis = ['a'] * 12
   *     >>> hyp_len = len(hypothesis)
   *     >>> closest_ref_len =  closest_ref_length(references, hyp_len)
   *     >>> brevity_penalty(closest_ref_len, hyp_len) # doctest: +ELLIPSIS
   *     0.9200...
   *
   * The brevity penalty doesn't depend on reference order. More importantly,
   * when two reference sentences are at the same distance, the shortest
   * reference sentence length is used.
   *
   *     >>> references = [['a'] * 13, ['a'] * 11]
   *     >>> hypothesis = ['a'] * 12
   *     >>> hyp_len = len(hypothesis)
   *     >>> closest_ref_len =  closest_ref_length(references, hyp_len)
   *     >>> bp1 = brevity_penalty(closest_ref_len, hyp_len)
   *     >>> hyp_len = len(hypothesis)
   *     >>> closest_ref_len =  closest_ref_length(reversed(references), hyp_len)
   *     >>> bp2 = brevity_penalty(closest_ref_len, hyp_len)
   *     >>> bp1 == bp2 == 1
   *     True
   *
   * A test example from mteval-v13a.pl (starting from the line 705):
   *
   *     >>> references = [['a'] * 11, ['a'] * 8]
   *     >>> hypothesis = ['a'] * 7
   *     >>> hyp_len = len(hypothesis)
   *     >>> closest_ref_len =  closest_ref_length(references, hyp_len)
   *     >>> brevity_penalty(closest_ref_len, hyp_len) # doctest: +ELLIPSIS
   *     0.8668...
   *
   *     >>> references = [['a'] * 11, ['a'] * 8, ['a'] * 6, ['a'] * 7]
   *     >>> hypothesis = ['a'] * 7
   *     >>> hyp_len = len(hypothesis)
   *     >>> closest_ref_len =  closest_ref_length(references, hyp_len)
   *     >>> brevity_penalty(closest_ref_len, hyp_len)
   *     1.0
   *
   * @param closestRefLen
   * @param hypLen
   * @return
   */
  def brevityPenalty(closestRefLen: Int, hypLen: Int): Double = {
    if (hypLen > closestRefLen) {
      1.0
    } else if (hypLen == 0) {
      0.0
    } else {
      math.exp(1 - closestRefLen.toDouble / hypLen.toDouble)
    }
  }

  /**
   * Calculate modified ngram precision.
   *
   * The normal precision method may lead to some wrong translations with
   * high-precision, e.g., the translation, in which a word of reference
   * repeats several times, has very high precision.
   *
   * This function only returns the Fraction object that contains the numerator
   * and denominator necessary to calculate the corpus-level precision.
   * To calculate the modified precision for a single pair of hypothesis and
   * references, cast the Fraction object into a float.
   *
   * The famous "the the the ... " example shows that you can get BLEU precision
   * by duplicating high frequency words.
   *
   *     reference1 = 'the cat is on the mat'.split()
   *     reference2 = 'there is a cat on the mat'.split()
   *     hypothesis1 = 'the the the the the the the'.split()
   *     references = [reference1, reference2]
   *     float(modifiedPrecision(references, hypothesis1, n=1))
   *     0.2857...
   *
   * In the modified n-gram precision, a reference word will be considered
   *    exhausted after a matching hypothesis word is identified, e.g.
   *
   *     reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
   *     ...               'ensures', 'that', 'the', 'military', 'will',
   *     ...               'forever', 'heed', 'Party', 'commands']
   *     reference2 = ['It', 'is', 'the', 'guiding', 'principle', 'which',
   *     ...               'guarantees', 'the', 'military', 'forces', 'always',
   *     ...               'being', 'under', 'the', 'command', 'of', 'the',
   *     ...               'Party']
   *     reference3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
   *     ...               'army', 'always', 'to', 'heed', 'the', 'directions',
   *     ...               'of', 'the', 'party']
   *     hypothesis = 'of the'.split()
   *     references = [reference1, reference2, reference3]
   *     float(modified_precision(references, hypothesis, n=1))
   *     1.0
   *     float(modified_precision(references, hypothesis, n=2))
   *     1.0
   *
   * An example of a normal machine translation hypothesis:
   *
   *     hypothesis1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
   *     ...               'ensures', 'that', 'the', 'military', 'always',
   *     ...               'obeys', 'the', 'commands', 'of', 'the', 'party']
   *
   *     hypothesis2 = ['It', 'is', 'to', 'insure', 'the', 'troops',
   *     ...               'forever', 'hearing', 'the', 'activity', 'guidebook',
   *     ...               'that', 'party', 'direct']
   *
   *     reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
   *     ...               'ensures', 'that', 'the', 'military', 'will',
   *     ...               'forever', 'heed', 'Party', 'commands']
   *
   *     reference2 = ['It', 'is', 'the', 'guiding', 'principle', 'which',
   *     ...               'guarantees', 'the', 'military', 'forces', 'always',
   *     ...               'being', 'under', 'the', 'command', 'of', 'the',
   *     ...               'Party']
   *
   *     reference3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
   *     ...               'army', 'always', 'to', 'heed', 'the', 'directions',
   *     ...               'of', 'the', 'party']
   *     references = [reference1, reference2, reference3]
   *     float(modified_precision(references, hypothesis1, n=1))
   *     0.9444...
   *     float(modified_precision(references, hypothesis2, n=1))
   *     0.5714...
   *     float(modified_precision(references, hypothesis1, n=2))
   *     0.5882352941176471
   *     float(modified_precision(references, hypothesis2, n=2))
   *     0.07692...
   *
   * @param references
   * @param hypothesis
   * @param n
   * @return
   */
  private def modifiedPrecision(
    references: Array[Array[String]],
    hypothesis: Array[String],
    n: Int): Fraction = {

    val counts = if (hypothesis.length > n) {
      counter[BoolArray](ngrams(hypothesis, n))
    } else {
      new mutable.HashMap[BoolArray, Int].withDefaultValue(0)
    }

    val keySet = counts.keySet
    val maxCounts = new mutable.HashMap[BoolArray, Int].withDefaultValue(0)
    var i = 0
    while (i < references.length) {
      val reference = references(i)
      val referenceCounts = if (reference.length >= n) {
        counter[BoolArray](ngrams(reference, n))
      } else {
        new mutable.HashMap[BoolArray, Int].withDefaultValue(0)
      }
      keySet.foreach(x => {
        val referCount = referenceCounts.getOrElse(x, 0)
        maxCounts.put(x,
          math.max(maxCounts.getOrElse(x, 0), referCount))
      })
      i += 1
    }
    val clippedCounts = new mutable.HashMap[BoolArray, Int].withDefaultValue(0)
    keySet.foreach(x => {
      clippedCounts.put(x,
        math.min(counts.get(x).get, maxCounts.get(x).get))
    })
    val numerator = clippedCounts.values.foldLeft(0)((a, b) => a + b)
    val denominator = math.max(1, counts.values.foldLeft(0)((a, b) => a + b))

    return Fraction.getFraction(numerator, denominator)
  }

  private def counter[A](key: Iterator[A]): mutable.Map[A, Int] = {
    val counts = new mutable.HashMap[A, Int]

    while (key.hasNext) {
      val _key = key.next
      counts(_key) = counts.getOrElse(_key, 0) + 1
    }
    counts
  }

  private def ngrams(sequence: Array[String], n: Int): Iterator[BoolArray] = {
    val seqIter = sequence.toIterator
    val ngram = new ArrayBuffer[BoolArray]
    var buffer = new mutable.MutableList[String]
    var count = n
    while (count > 1) {
      buffer += seqIter.next
      count -= 1
    }
    while (seqIter.hasNext) {
      val _next = seqIter.next
      buffer += _next
      ngram.append(new BoolArray(buffer.toArray))
      buffer = buffer.drop(1)
    }

    ngram.toIterator
  }

  def closestRefLength(
    references: Array[Array[String]],
    hypLen: Int): Int = {
    references.map(x => (x.length, math.abs(hypLen - x.length))).minBy(_._2)._1
  }
}

class BoolArray(val array: Array[String])
  extends Serializable {

  override def equals(obj: Any): Boolean = {
    if (!obj.isInstanceOf[BoolArray]) {
      false
    } else {
      val other = obj.asInstanceOf[BoolArray]
      if (array.length != other.array.length) {
        return false
      }
      array zip other.array foreach(x => {
        if (!x._1.equals(x._2)) {
          return false
        }
      })
      true
    }
  }

  override def hashCode(): Int = {
    var hash = 0
    array.foreach(x => hash += x.hashCode)
    hash
  }
}

case class SmoothingFunction(epsilon: Double = 0.1, alpha: Int = 5, k: Int = 5)
  extends Serializable {

  /**
   * No smoothing
   * @param pN
   * @return
   */
  def method0()(pN: Array[Fraction]): Array[Fraction] = {
    val pNsmooth = new ArrayBuffer[Fraction]
    pN.foreach(x => {
      if (x.getNumerator != 0) {
        pNsmooth.append(Fraction.getFraction(x.getNumerator, x.getDenominator))
      }
    })
    pNsmooth.toArray
  }

  /**
   * Smoothing method 1: Add *epsilon* counts to precision with 0 counts.
   * @param pN
   * @return
   */
  def method1(pN: Array[Fraction]): Array[Fraction] = {
    pN.map(x => {
      if (x.getNumerator == 0) {
        Fraction.getFraction(x.getNumerator + epsilon / x.getDenominator)
      } else {
        x
      }
    })
  }

  /**
   * Smoothing method 2: Add 1 to both numerator and denominator from
   *     Chin-Yew Lin and Franz Josef Och (2004) Automatic evaluation of
   *     machine translation quality using longest common subsequence and
   *     skip-bigram statistics. In ACL04.
   * @param pN
   * @return
   */
  def method2(pN: Array[Fraction]): Array[Fraction] = {
    pN.map(x => Fraction.getFraction(x.getNumerator + 1 / x.getDenominator + 1))
  }

  /**
   * Smoothing method 3: NIST geometric sequence smoothing
   *     The smoothing is computed by taking 1 / ( 2^k ), instead of 0, for each
   *     precision score whose matching n-gram count is null.
   *     k is 1 for the first 'n' value for which the n-gram match count is null/
   *     For example, if the text contains:
   *      - one 2-gram match
   *      - and (consequently) two 1-gram matches
   *     the n-gram count for each individual precision score would be:
   * - n=1  =>  prec_count = 2     (two unigrams)
   * - n=2  =>  prec_count = 1     (one bigram)
   * - n=3  =>  prec_count = 1/2   (no trigram,  taking 'smoothed' value of 1 / ( 2^k ), with k=1)
   * - n=4  =>  prec_count = 1/4   (no fourgram, taking 'smoothed' value of 1 / ( 2^k ), with k=2)
   * @param pN
   * @return
   */
  def method3(pN: Array[Fraction]): Array[Fraction] = {
    var incvnt = 0
    pN.map(x => {
      if (x.getNumerator == 0) {
        incvnt += 1
        Fraction.getFraction(1 / (math.pow(2, incvnt) * x.getDenominator))
      } else {
        x
      }
    })
  }
}
