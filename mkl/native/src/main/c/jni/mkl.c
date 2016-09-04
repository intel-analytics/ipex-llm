#include <jni.h>
#include <omp.h>

#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_intel_webscaleml_mkl_MKL
 * Method:    setNumThreads
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_com_intel_webscaleml_mkl_MKL_setNumThreads
  (JNIEnv * env, jclass cls, jint num_threads) {

  omp_set_num_threads(num_threads);
}


/*
 * Class:     com_intel_webscaleml_mkl_MKL
 * Method:    getNumThreads
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_intel_webscaleml_mkl_MKL_getNumThreads
  (JNIEnv * env, jclass cls) {
  return omp_get_max_threads();
}

#ifdef __cplusplus
}
#endif