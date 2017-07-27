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
   * sentenceBleu([ref1, ref2, ref3], hypo1)
   *
   * The default BLEU will calculate a score using 4-grams with uniformly
   * distributed weights. You can define your customized weights by transferring
   * the arguments to weights.
   *
   * E.g:
   *  weights = Array(0.2, 0.2, 0.2, 0.2, 0.2)
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
    smoothingFunction: (Array[Fraction], Array[Array[String]], Array[String], Int, Boolean)
      => Array[Fraction] = null,
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
   * Calculate a system-level BLEU for all
   * the hypotheses and their respective references.
   *
   * listOfReferences = [[ref1.1, ref1.2, ref1.3], [ref2.1]]
   * hypotheses = [hyp1, hyp2]
   * corpus_bleu(listOfReferences, hypotheses)
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
    smoothingFunction:
    (Array[Fraction], Array[Array[String]], Array[String], Int, Boolean)
      => Array[Fraction] = null,
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

    val pNsmooth = _smoothingFunction(
      pN, references.last, hypotheses.last, hypLengths, emulateMultibleu)
    val s = _weights.zip(pNsmooth)
      .map(x => x._1 * math.log(x._2.doubleValue))
      .foldLeft(0.0)((a, b) => a + b)
    val s_sum = bp * math.exp(s)

    s_sum
  }

  /**
   * Calculate brevity penalty.
   *
   * It will give penalty for short candidate sentence for the BLEU score.
   *
   * For example,
   * references = Array(Array.fill(10)("a"), Array.fill(12)("a"))
   * hypothesis = Array.fill(15)("b")
   *
   * will return
   *
   * exp(1 - 15 / 12)
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

  def modifiedPrecision(
    references: Array[Array[String]],
    hypothesis: Array[String],
    n: Int): Fraction = {
    BLEU.modifiedPrecision(references, hypothesis, n)
  }

  private def counter[A](key: Iterator[A]): mutable.Map[A, Int] = {
    BLEU.counter(key)
  }

  private def ngrams(sequence: Array[String], n: Int): Iterator[BoolArray] = {
    BLEU.ngrams(sequence, n)
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
  def method0()(pN: Array[Fraction],
                refs: Array[Array[String]] = null,
                hypo: Array[String] = null,
                hypLen: Int = 0,
                emuMultibleu: Boolean = false): Array[Fraction] = {
    val pNsmooth = new ArrayBuffer[Fraction]
    var i = 0
    pN.foreach(x => {
      if (x.getNumerator != 0) {
        pNsmooth.append(Fraction.getFraction(x.getNumerator, x.getDenominator))
      } else if (emuMultibleu == true && i < 5) {
        return Array(Fraction.getFraction(Float.MinValue))
      }
      i += 1
    })
    pNsmooth.toArray
  }

  /**
   * Smoothing method 1: Add *epsilon* counts to precision with 0 counts.
   * @param pN
   * @return
   */
  def method1()(pN: Array[Fraction],
                refs: Array[Array[String]] = null,
                hypo: Array[String] = null,
                hypLen: Int = 0,
                emuMultibleu: Boolean = false): Array[Fraction] = {
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
  def method2()(pN: Array[Fraction],
                refs: Array[Array[String]] = null,
                hypo: Array[String] = null,
                hypLen: Int = 0,
                emuMultibleu: Boolean = false): Array[Fraction] = {
    pN.map(x => Fraction.getFraction(x.getNumerator + 1, x.getDenominator + 1))
  }

  /**
   * Smoothing method 3: NIST geometric sequence smoothing
   *     The smoothing is computed by taking 1 / ( 2^k ), instead of 0, for each
   *     precision score whose matching n-gram count is null.
   * @param pN
   * @return
   */
  def method3()(pN: Array[Fraction],
                refs: Array[Array[String]] = null,
                hypo: Array[String] = null,
                hypLen: Int = 0,
                emuMultibleu: Boolean = false): Array[Fraction] = {
    var incvnt = 0
    pN.map(x => {
      if (x.getNumerator == 0) {
        incvnt += 1
        Fraction.getFraction(1.0 / (math.pow(2, incvnt) * x.getDenominator.toDouble))
      } else {
        x
      }
    })
  }

  /**
   *  Smoothing method 4:
   *     Shorter translations may have inflated precision values due to having
   *     smaller denominators; therefore, we give them proportionally
   *     smaller smoothed counts. Instead of scaling to 1/(2^k), Chen and Cherry
   *     suggests dividing by 1/ln(len(T)), where T is the length of the translation.
   * @param pN
   * @param refs
   * @param hypo
   * @param hypLen
   * @param emuMultibleu
   * @return
   */
  def method4()(pN: Array[Fraction],
                refs: Array[Array[String]] = null,
                hypo: Array[String] = null,
                hypLen: Int = 0,
                emuMultibleu: Boolean = false): Array[Fraction] = {
    var i = -1
    pN.map(x => {
      i += 1
      if (x.getNumerator == 0 && hypLen != 0) {
        val incvnt = i + 1.0 * k.toDouble / math.log(hypLen)
        Fraction.getFraction(1.0 / incvnt)
      } else {
        x
      }
    })
  }

  /**
   * Smoothing method 5:
   *     The matched counts for similar values of n should be similar. To a
   *     calculate the n-gram matched count, it averages the n−1, n and n+1 gram
   *     matched counts.
   * @param pN
   * @param refs
   * @param hypo
   * @param hypLen
   * @param emuMultibleu
   * @return
   */
  def method5()(pN: Array[Fraction],
                refs: Array[Array[String]] = null,
                hypo: Array[String] = null,
                hypLen: Int = 0,
                emuMultibleu: Boolean = false): Array[Fraction] = {
    val pNPlus1 = pN ++ Array(BLEU.modifiedPrecision(references = refs, hypothesis = hypo, n = 5))
    var m = pN.head.doubleValue + 1.0
    val pNsmooth = new ArrayBuffer[Fraction]
    var i = 0
    pN.map(x => {
      val tmp = (m + x.doubleValue + pNPlus1(i + 1).doubleValue) / 3.0
      m = tmp
      i += 1
      Fraction.getFraction(tmp)
    })
  }

  /**
   *    Smoothing method 6:
   *    Interpolates the maximum likelihood estimate of the precision *p_n* with
   *     a prior estimate *pi0*. The prior is estimated by assuming that the ratio
   *     between pn and pn−1 will be the same as that between pn−1 and pn−2; from
   *     Gao and He (2013) Training MRF-Based Phrase Translation Models using
   *     Gradient Ascent. In NAACL.
   * @param pN
   * @param refs
   * @param hypo
   * @param hypLen
   * @param emuMultibleu
   * @return
   */
  def method6()(pN: Array[Fraction],
                refs: Array[Array[String]] = null,
                hypo: Array[String] = null,
                hypLen: Int = 0,
                emuMultibleu: Boolean = false): Array[Fraction] = {
    require(pN.length > 2)
    val pNsmooth = new ArrayBuffer[Fraction]
    var i = 0
    while (i < pN.length) {
      if (i < 2) {
        pNsmooth.append(pN(i))
      } else {
        val pi0 = if (pNsmooth(i - 2).doubleValue == 0) {
          0.0
        } else {
          math.pow(pNsmooth(i - 1).doubleValue, 2) / pNsmooth(i - 2).doubleValue
        }
        val m = pN(i).getNumerator.toDouble
        val l = BLEU.ngrams(hypo, i + 1).toArray.length.toDouble
        pNsmooth.append(Fraction.getFraction((m + alpha.toDouble * pi0) / (l + alpha.toDouble)))
      }
      i += 1
    }
    pNsmooth.toArray
  }

  /**
   * Smoothing method 6:
   *     Interpolates the maximum likelihood estimate of the precision *p_n* with
   *     a prior estimate *pi0*. The prior is estimated by assuming that the ratio
   *     between pn and pn−1 will be the same as that between pn−1 and pn−2.
   * @param pN
   * @param refs
   * @param hypo
   * @param hypLen
   * @param emuMultibleu
   * @return
   */
  def method7()(pN: Array[Fraction],
                refs: Array[Array[String]] = null,
                hypo: Array[String] = null,
                hypLen: Int = 0,
                emuMultibleu: Boolean = false): Array[Fraction] = {
    val pN1 = method4()(pN, refs, hypo, hypLen, emuMultibleu)
    val pN2 = method5()(pN1, refs, hypo, hypLen, emuMultibleu)
    pN2
  }
}

object BLEU {

  def apply(): BLEU = new BLEU()

  def ngrams(sequence: Array[String], n: Int): Iterator[BoolArray] = {
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

  /**
   * Calculate modified ngram precision.
   *
   * It will element-wisely compare the n-gram phrases
   * from candidate sentences and reference sentences
   * and return count number for the pair phrases.
   *
   * The number of paired phrases will be further clipped
   * for giving penalty to duplicated phrases.
   * @param references
   * @param hypothesis
   * @param n
   * @return
   */
  def modifiedPrecision(
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
}
