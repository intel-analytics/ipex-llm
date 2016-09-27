#include <jni.h>
#include <omp.h>
#include <mkl.h>

#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_intel_webscaleml_mkl_MKL
 * Method:    setNumThreads
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_setNumThreads
  (JNIEnv * env, jclass cls, jint num_threads) {
  omp_set_num_threads(num_threads);
}


/*
 * Class:     com_intel_webscaleml_mkl_MKL
 * Method:    getNumThreads
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_getNumThreads
  (JNIEnv * env, jclass cls) {
  return omp_get_max_threads();
}

/*
 * Class:     com_intel_analytics_sparkdl_mkl_MKL
 * Method:    vsPowx
 * Signature: (I[FIF[FI)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_vsPowx
  (JNIEnv * env, jclass cls, jint n, jfloatArray a, jint aOffset, jfloat b, jfloatArray y,
  jint yOffset) {

 jfloat * jni_a = (*env)->GetPrimitiveArrayCritical(env, a, JNI_FALSE);
 jfloat * jni_y = (*env)->GetPrimitiveArrayCritical(env, y, JNI_FALSE);

 vsPowx( n, jni_a + aOffset, b, jni_y + yOffset);

 (*env)->ReleasePrimitiveArrayCritical(env, y, jni_y, 0);
 (*env)->ReleasePrimitiveArrayCritical(env, a, jni_a, 0);
}

 /*
  * Class:     com_intel_analytics_sparkdl_mkl_MKL
  * Method:    vdPowx
  * Signature: (I[DID[DI)V
  */
JNIEXPORT void JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_vdPowx
   (JNIEnv * env, jclass cls, jint n, jdoubleArray a, jint aOffset, jdouble b, jdoubleArray y,
   jint yOffset) {

   jdouble * jni_a = (*env)->GetPrimitiveArrayCritical(env, a, JNI_FALSE);
   jdouble * jni_y = (*env)->GetPrimitiveArrayCritical(env, y, JNI_FALSE);

   vdPowx( n, jni_a + aOffset, b, jni_y + yOffset);

   (*env)->ReleasePrimitiveArrayCritical(env, y, jni_y, 0);
   (*env)->ReleasePrimitiveArrayCritical(env, a, jni_a, 0);
 }

/*
 * Class:     com_intel_analytics_sparkdl_mkl_MKL
 * Method:    vsLn
 * Signature: (I[FI[FI)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_vsLn
  (JNIEnv * env, jclass cls, jint n, jfloatArray a, jint aOffset, jfloatArray y,
  jint yOffset) {

 jfloat * jni_a = (*env)->GetPrimitiveArrayCritical(env, a, JNI_FALSE);
 jfloat * jni_y = (*env)->GetPrimitiveArrayCritical(env, y, JNI_FALSE);

 vsLn( n, jni_a + aOffset, jni_y + yOffset);

 (*env)->ReleasePrimitiveArrayCritical(env, y, jni_y, 0);
 (*env)->ReleasePrimitiveArrayCritical(env, a, jni_a, 0);
}

 /*
  * Class:     com_intel_analytics_sparkdl_mkl_MKL
  * Method:    vdLn
  * Signature: (I[DI[DI)V
  */
JNIEXPORT void JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_vdLn
   (JNIEnv * env, jclass cls, jint n, jdoubleArray a, jint aOffset, jdoubleArray y,
   jint yOffset) {

   jdouble * jni_a = (*env)->GetPrimitiveArrayCritical(env, a, JNI_FALSE);
   jdouble * jni_y = (*env)->GetPrimitiveArrayCritical(env, y, JNI_FALSE);

   vdLn( n, jni_a + aOffset, jni_y + yOffset);

   (*env)->ReleasePrimitiveArrayCritical(env, y, jni_y, 0);
   (*env)->ReleasePrimitiveArrayCritical(env, a, jni_a, 0);
 }

 /*
  * Class:     com_intel_analytics_sparkdl_mkl_MKL
  * Method:    vsExp
  * Signature: (I[FI[FI)V
  */
 JNIEXPORT void JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_vsExp
   (JNIEnv * env, jclass cls, jint n, jfloatArray a, jint aOffset, jfloatArray y,
   jint yOffset) {

  jfloat * jni_a = (*env)->GetPrimitiveArrayCritical(env, a, JNI_FALSE);
  jfloat * jni_y = (*env)->GetPrimitiveArrayCritical(env, y, JNI_FALSE);

  vsExp( n, jni_a + aOffset, jni_y + yOffset);

  (*env)->ReleasePrimitiveArrayCritical(env, y, jni_y, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, a, jni_a, 0);
 }

  /*
   * Class:     com_intel_analytics_sparkdl_mkl_MKL
   * Method:    vdExp
   * Signature: (I[DI[DI)V
   */
 JNIEXPORT void JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_vdExp
    (JNIEnv * env, jclass cls, jint n, jdoubleArray a, jint aOffset, jdoubleArray y,
    jint yOffset) {

    jdouble * jni_a = (*env)->GetPrimitiveArrayCritical(env, a, JNI_FALSE);
    jdouble * jni_y = (*env)->GetPrimitiveArrayCritical(env, y, JNI_FALSE);

    vdExp( n, jni_a + aOffset, jni_y + yOffset);

    (*env)->ReleasePrimitiveArrayCritical(env, y, jni_y, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, a, jni_a, 0);
  }

  /*
   * Class:     com_intel_analytics_sparkdl_mkl_MKL
   * Method:    vsSqrt
   * Signature: (I[FI[FI)V
   */
  JNIEXPORT void JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_vsSqrt
    (JNIEnv * env, jclass cls, jint n, jfloatArray a, jint aOffset, jfloatArray y,
    jint yOffset) {

   jfloat * jni_a = (*env)->GetPrimitiveArrayCritical(env, a, JNI_FALSE);
   jfloat * jni_y = (*env)->GetPrimitiveArrayCritical(env, y, JNI_FALSE);

   vsSqrt( n, jni_a + aOffset, jni_y + yOffset);

   (*env)->ReleasePrimitiveArrayCritical(env, y, jni_y, 0);
   (*env)->ReleasePrimitiveArrayCritical(env, a, jni_a, 0);
  }

   /*
    * Class:     com_intel_analytics_sparkdl_mkl_MKL
    * Method:    vdSqrt
    * Signature: (I[DI[DI)V
    */
  JNIEXPORT void JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_vdSqrt
     (JNIEnv * env, jclass cls, jint n, jdoubleArray a, jint aOffset, jdoubleArray y,
     jint yOffset) {

     jdouble * jni_a = (*env)->GetPrimitiveArrayCritical(env, a, JNI_FALSE);
     jdouble * jni_y = (*env)->GetPrimitiveArrayCritical(env, y, JNI_FALSE);

     vdSqrt( n, jni_a + aOffset, jni_y + yOffset);

     (*env)->ReleasePrimitiveArrayCritical(env, y, jni_y, 0);
     (*env)->ReleasePrimitiveArrayCritical(env, a, jni_a, 0);
   }

   /*
    * Class:     com_intel_analytics_sparkdl_mkl_MKL
    * Method:    vsLog1p
    * Signature: (I[FI[FI)V
    */
   JNIEXPORT void JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_vsLog1p
     (JNIEnv * env, jclass cls, jint n, jfloatArray a, jint aOffset, jfloatArray y,
     jint yOffset) {

    jfloat * jni_a = (*env)->GetPrimitiveArrayCritical(env, a, JNI_FALSE);
    jfloat * jni_y = (*env)->GetPrimitiveArrayCritical(env, y, JNI_FALSE);

    vsLog1p( n, jni_a + aOffset, jni_y + yOffset);

    (*env)->ReleasePrimitiveArrayCritical(env, y, jni_y, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, a, jni_a, 0);
   }

    /*
     * Class:     com_intel_analytics_sparkdl_mkl_MKL
     * Method:    vdLog1p
     * Signature: (I[DI[DI)V
     */
   JNIEXPORT void JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_vdLog1p
      (JNIEnv * env, jclass cls, jint n, jdoubleArray a, jint aOffset, jdoubleArray y,
      jint yOffset) {

      jdouble * jni_a = (*env)->GetPrimitiveArrayCritical(env, a, JNI_FALSE);
      jdouble * jni_y = (*env)->GetPrimitiveArrayCritical(env, y, JNI_FALSE);

      vdLog1p( n, jni_a + aOffset, jni_y + yOffset);

      (*env)->ReleasePrimitiveArrayCritical(env, y, jni_y, 0);
      (*env)->ReleasePrimitiveArrayCritical(env, a, jni_a, 0);
    }

 /*
  * Class:     com_intel_analytics_sparkdl_mkl_MKL
  * Method:    vsMul
  * Signature: (I[FI[FI[FI)V
  */
JNIEXPORT void JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_vsMul
   (JNIEnv * env, jclass cls, jint n, jfloatArray a, jint aOffset, jfloatArray b,
   jint bOffset, jfloatArray y, jint yOffset) {

   jfloat * jni_a = (*env)->GetPrimitiveArrayCritical(env, a, JNI_FALSE);
   jfloat * jni_b = (*env)->GetPrimitiveArrayCritical(env, b, JNI_FALSE);
   jfloat * jni_y = (*env)->GetPrimitiveArrayCritical(env, y, JNI_FALSE);

   vsMul( n, jni_a + aOffset, jni_b + bOffset, jni_y + yOffset);

   (*env)->ReleasePrimitiveArrayCritical(env, y, jni_y, 0);
   (*env)->ReleasePrimitiveArrayCritical(env, b, jni_b, 0);
   (*env)->ReleasePrimitiveArrayCritical(env, a, jni_a, 0);
 }

 /*
  * Class:     com_intel_analytics_sparkdl_mkl_MKL
  * Method:    vdMul
  * Signature: (I[DI[DI[DI)V
  */
JNIEXPORT void JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_vdMul
   (JNIEnv * env, jclass cls, jint n, jdoubleArray a, jint aOffset, jdoubleArray b,
   jint bOffset, jdoubleArray y, jint yOffset) {

   jdouble * jni_a = (*env)->GetPrimitiveArrayCritical(env, a, JNI_FALSE);
   jdouble * jni_b = (*env)->GetPrimitiveArrayCritical(env, b, JNI_FALSE);
   jdouble * jni_y = (*env)->GetPrimitiveArrayCritical(env, y, JNI_FALSE);

   vdMul( n, jni_a + aOffset, jni_b + bOffset, jni_y + yOffset);

   (*env)->ReleasePrimitiveArrayCritical(env, y, jni_y, 0);
   (*env)->ReleasePrimitiveArrayCritical(env, b, jni_b, 0);
   (*env)->ReleasePrimitiveArrayCritical(env, a, jni_a, 0);
}


/*
 * Class:     com_intel_analytics_sparkdl_mkl_MKL
 * Method:    vsDiv
 * Signature: (I[FI[FI[FI)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_vsDiv
  (JNIEnv * env, jclass cls, jint n, jfloatArray a, jint aOffset, jfloatArray b, jint bOffset,
  jfloatArray y, jint yOffset) {


   jfloat * jni_a = (*env)->GetPrimitiveArrayCritical(env, a, JNI_FALSE);
   jfloat * jni_b = (*env)->GetPrimitiveArrayCritical(env, b, JNI_FALSE);
   jfloat * jni_y = (*env)->GetPrimitiveArrayCritical(env, y, JNI_FALSE);

   vsDiv(n, jni_a + aOffset, jni_b + bOffset, jni_y + yOffset);

   (*env)->ReleasePrimitiveArrayCritical(env, y, jni_y, 0);
   (*env)->ReleasePrimitiveArrayCritical(env, b, jni_b, 0);
   (*env)->ReleasePrimitiveArrayCritical(env, a, jni_a, 0);
 }

/*
 * Class:     com_intel_analytics_sparkdl_mkl_MKL
 * Method:    vdDiv
 * Signature: (I[DI[DI[DI)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_vdDiv
  (JNIEnv * env, jclass cls, jint n, jfloatArray a, jint aOffset, jfloatArray b, jint bOffset,
  jfloatArray y, jint yOffset) {


   jdouble * jni_a = (*env)->GetPrimitiveArrayCritical(env, a, JNI_FALSE);
   jdouble * jni_b = (*env)->GetPrimitiveArrayCritical(env, b, JNI_FALSE);
   jdouble * jni_y = (*env)->GetPrimitiveArrayCritical(env, y, JNI_FALSE);

   vdDiv(n, jni_a + aOffset, jni_b + bOffset, jni_y + yOffset);

   (*env)->ReleasePrimitiveArrayCritical(env, y, jni_y, 0);
   (*env)->ReleasePrimitiveArrayCritical(env, b, jni_b, 0);
   (*env)->ReleasePrimitiveArrayCritical(env, a, jni_a, 0);
}

#ifdef __cplusplus
}
#endif
