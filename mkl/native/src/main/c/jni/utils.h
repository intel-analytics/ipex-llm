#ifndef _UTILS_H_
#define _UTILS_H_

#include "cpu_info.hpp"

int computeOut(int input, int pad, int kernle, int stride,
               bool ceilMode = false);

#include <omp.h>
#include <sched.h>

template <typename DType>
void setValue(const int N, const DType alpha, DType* Y) {
  // If we are executing parallel region already then do not start another one
  // if also number of data to be processed is smaller than arbitrary:
  // threashold 12*4 cachelines per thread then no parallelization is to be made
  #ifdef _OPENMP

  int nthr = omp_get_max_threads();
  int threshold = nthr * caffe::cpu::OpenMpManager::getProcessorSpeedMHz() / 3;
  bool run_parallel =  // Do not do parallel computation from non major threads
       caffe::cpu::OpenMpManager::isMajorThread(std::this_thread::get_id());

  // Note: we Assume GPU's CPU path is single threaded
  if (omp_in_parallel() == 0) {
    // inactive parallel region may mean also batch 1,
    // but no new threads are to be created
    run_parallel = run_parallel && (N >= threshold);
  } else {
    // If we are running active parallel region then it is CPU
    run_parallel = run_parallel && (N >= threshold);
  }

  if (run_parallel) {
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
      Y[i] = alpha;
    }

    return;
  }

  #endif

  if (alpha == 0) {
    memset(Y, 0, sizeof(DType) * N);  // NOLINT(caffe/alt_fn)
  } else {
    std::fill(Y, Y + N, alpha);
  }
}


#endif
