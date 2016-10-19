#include <jni.h>

#include "debug.h"
#include "layer.h"
#include "memory.h"
#include "utils.h"

template <typename DType>
class MKLLRN : public MKLLayer<DType>
{
 public:
  MKLLRN();
  ~MKLLRN();

  void init(size_t inputNumber, size_t inputChannel, size_t inputHeight,
            size_t inputWidth, int size, DType alpha, DType beta, DType k,
            int dimension);

  void updateOutput(DType *input, DType *output);
  void updateGradInput(DType *input, DType *gradOutput, DType *gradInput);

 private:
  // this method is not the same as createMklLayout in MKLMemory
  void firstPass();
  void preExecute(DType *input);

  std::shared_ptr<MKLData<DType>> workspace;

  int size;
  DType alpha;
  DType beta;
  DType k;

  size_t inputSize[4];
  size_t inputStrides[4];

  size_t outputSize[4];
  size_t outputStrides[4];
};

template <typename DType>
MKLLRN<DType>::MKLLRN() : workspace(new MKLData<DType>)
{
}

template <typename DType>
MKLLRN<DType>::~MKLLRN()
{
}

template <typename DType>
void MKLLRN<DType>::init(size_t inputNumber, size_t inputChannel,
                         size_t inputHeight, size_t inputWidth, int size,
                         DType alpha, DType beta, DType k, int dimension)
{
  this->dimension = dimension;

  inputSize[0] = inputWidth;
  inputSize[1] = inputHeight;
  inputSize[2] = inputChannel;
  inputSize[3] = inputNumber;

  inputStrides[0] = 1;
  for (int i        = 1; i < 4; i++)
    inputStrides[i] = inputStrides[i - 1] * inputSize[i - 1];

  // the output channel is as same as the number of kernel.
  // and the output number must be as same as the number of input too.
  outputSize[0] = inputWidth;
  outputSize[1] = inputHeight;
  outputSize[2] = inputChannel;
  outputSize[3] = inputNumber;

  outputStrides[0] = 1;
  for (int i         = 1; i < 4; i++)
    outputStrides[i] = outputStrides[i - 1] * outputSize[i - 1];

  this->size  = size;
  this->alpha = alpha;
  this->beta  = beta;
  this->k     = k;

  // create usr layout
  this->input->createUsrLayout(dimension, inputSize, inputStrides);
  this->output->createUsrLayout(dimension, outputSize, outputStrides);

  this->gradInput->createUsrLayout(dimension, inputSize, inputStrides);
  this->gradOutput->createUsrLayout(dimension, outputSize, outputStrides);
}

template <typename DType>
void MKLLRN<DType>::firstPass()
{
  dnnError_t status = E_UNIMPLEMENTED;
  dnnLayout_t layout = NULL;

  if (this->input->isUsePrev()) {
    layout = this->input->layoutPrev;
  }
  if (!layout) {
    status =
      dnnLayoutCreate<DType>(&layout, this->dimension, inputSize, inputStrides);
    CHECK_EQ(status, E_SUCCESS);
  }

  status = dnnLRNCreateForward<DType>(&(this->forwardPrim), NULL, layout, size,
                                      alpha, beta, k);
  CHECK_EQ(status, E_SUCCESS);

  this->input->createMklLayout(this->forwardPrim, dnnResourceSrc);
  this->output->createMklLayout(this->forwardPrim, dnnResourceDst);

  status = dnnLRNCreateBackward<DType>(&(this->backwardPrim), NULL, layout,
                                       layout, size, alpha, beta, k);
  CHECK_EQ(status, E_SUCCESS);

  this->gradOutput->createMklLayout(this->backwardPrim, dnnResourceDiffDst);
  this->gradInput->createMklLayout(this->backwardPrim, dnnResourceDiffSrc);

  // create workspace
  this->workspace->createMklLayout(this->forwardPrim, dnnResourceWorkspace);
  this->workspace->createConversion(true);

  if (!this->input->isUsePrev()) {
    dnnLayoutDelete<DType>(layout);
  }

  // we create the layout only at the first time
  this->isFirstPass = false;
}

template <typename DType>
void MKLLRN<DType>::preExecute(DType *input)
{
  caffe::cpu::OpenMpManager::setGpuDisabled();
  caffe::cpu::OpenMpManager::bindOpenMpThreads();

  this->input->createConversion();
}

template <typename DType>
void MKLLRN<DType>::updateOutput(DType *input, DType *output)
{
  caffe::cpu::OpenMpManager::setGpuDisabled();
  caffe::cpu::OpenMpManager::bindOpenMpThreads();

  if (this->isFirstPass) firstPass();

  // Because the address will change every time, so we need create conversion
  // every forward/backward.
  // TODO Should we set the kernel and bias address every time?
  preExecute(input);
  this->output->createConversion();
  // this->output->setZero();
  this->workspace->setZero();

#ifdef DEBUG
  printData<DType>(reinterpret_cast<DType *>(this->input->getUsrData()),
                   this->inputSize[3], this->inputSize[2], this->inputSize[1],
                   this->inputSize[0], "Forward input");
#endif

  dnnError_t status;
  void *resources[dnnResourceNumber];

  resources[dnnResourceSrc]       = this->input->getConvertedData();
  resources[dnnResourceDst]       = this->output->getData();
  resources[dnnResourceWorkspace] = this->workspace->getData();

  PERFSTART();
  status = dnnExecute<DType>(this->forwardPrim, resources);
  PERFEND("main computing");
  CHECK_EQ(status, E_SUCCESS);

#ifdef DEBUG
  printData<DType>(reinterpret_cast<DType *>(this->output->getData()),
                   outputSize[3], outputSize[2], outputSize[1], outputSize[0],
                   "Forward output");
#endif

  if (!this->output->isUseNext()) {
    this->output->backToUsr();
  }
}

template <typename DType>
void MKLLRN<DType>::updateGradInput(DType *input, DType *gradOutput,
                                    DType *gradInput)
{
  caffe::cpu::OpenMpManager::setGpuDisabled();
  caffe::cpu::OpenMpManager::bindOpenMpThreads();

  dnnError_t status;
  void *resources[dnnResourceNumber];

  preExecute(input);

  this->gradOutput->createConversion();
  this->gradInput->createConversion();

  resources[dnnResourceDiffDst]   = this->gradOutput->getConvertedData();
  resources[dnnResourceDiffSrc]   = this->gradInput->getData();
  resources[dnnResourceSrc]       = this->input->getConvertedData();
  resources[dnnResourceWorkspace] = this->workspace->getData();

  // 4. main computing parts.
  PERFSTART();
  status = dnnExecute<DType>(this->backwardPrim, resources);
  CHECK_EQ(status, E_SUCCESS);
  PERFEND("main computing");

  if (!this->gradInput->isUsePrev()) {
    this->gradInput->backToUsr();
  }

#ifdef DEBUG
  printData<DType>(reinterpret_cast<DType *>(this->gradInput->getUsrData()),
                   inputSize[3], inputSize[2], inputSize[1], inputSize[0],
                   "backward gradient input");
#endif
}

template <typename ArrayType, typename DType>
jlong JNILRNInit(JNIEnv *env, jclass thisClass, jint inputNumber,
                 jint inputChannel, jint inputHeight, jint inputWidth,
                 jint size, DType alpha, DType beta, DType k, jint dimension)
{
  MKLLRN<DType> *lrn = new MKLLRN<DType>();
  lrn->init(inputNumber, inputChannel, inputHeight, inputWidth, size, alpha,
            beta, k, dimension);

  return reinterpret_cast<long>(lrn);
}

template <typename ArrayType, typename DType>
void JNILRNUpdateOutput(JNIEnv *env, jclass thisClass, ArrayType input,
                        jint inputOffset, ArrayType output, jint outputOffset,
                        long classPtr)
{
  MKLLRN<DType> *ptr = reinterpret_cast<MKLLRN<DType> *>(classPtr);

  std::shared_ptr<ZipArray<ArrayType, DType>> jInput(
      new ZipArray<ArrayType, DType>(env, input, inputOffset, ptr->input));

  std::shared_ptr<ZipArray<ArrayType, DType>> jOutput(
      new ZipArray<ArrayType, DType>(env, output, outputOffset, ptr->output));

  ptr->updateOutput(jInput->getPtr(), jOutput->getPtr());
}

template <typename ArrayType, typename DType>
void JNILRNUpdateGradInput(JNIEnv *env, jclass thisClass, ArrayType input,
                           jint inputOffset, ArrayType outputDiff,
                           jint outputDiffOffset, ArrayType inputDiff,
                           jint inputDiffOffset, long classPtr)
{
  MKLLRN<DType> *ptr = reinterpret_cast<MKLLRN<DType> *>(classPtr);
  std::shared_ptr<ZipArray<ArrayType, DType>> jInput(
      new ZipArray<ArrayType, DType>(env, input, inputOffset, ptr->input));

  std::shared_ptr<ZipArray<ArrayType, DType>> jOutputDiff(
      new ZipArray<ArrayType, DType>(env, outputDiff, outputDiffOffset,
                                     ptr->gradOutput));

  std::shared_ptr<ZipArray<ArrayType, DType>> jInputDiff(
      new ZipArray<ArrayType, DType>(env, inputDiff, inputDiffOffset,
                                     ptr->gradInput));

  ptr->updateGradInput(jInput->getPtr(), jOutputDiff->getPtr(),
                       jInputDiff->getPtr());
}

// Macro
#define LRNInit(DType, JType, JArrayType)                                    \
  JNIEXPORT                                                                  \
  jlong JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_LRNInit##DType(     \
      JNIEnv *env, jclass thisClass, jint inputNumber, jint inputChannel,    \
      jint inputHeight, jint inputWidth, jint size, JType alpha, JType beta, \
      JType k, jint dimension)                                               \
  {                                                                          \
    return JNILRNInit<JArrayType, JType>(                                    \
        env, thisClass, inputNumber, inputChannel, inputHeight, inputWidth,  \
        size, alpha, beta, k, dimension);                                    \
  }

#define LRNForward(DType, JType, JArrayType)                                  \
  JNIEXPORT                                                                   \
  void JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_LRNForward##DType(    \
      JNIEnv *env, jclass thisClass, JArrayType input, jint inputOffset,      \
      JArrayType output, jint outputOffset, long classPtr)                    \
  {                                                                           \
    JNILRNUpdateOutput<JArrayType, JType>(env, thisClass, input, inputOffset, \
                                          output, outputOffset, classPtr);    \
  }

#define LRNBackward(DType, JType, JArrayType)                               \
  JNIEXPORT                                                                 \
  void JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_LRNBackward##DType( \
      JNIEnv *env, jclass thisClass, JArrayType input, jint inputOffset,    \
      JArrayType outputDiff, jint outputDiffOffset, JArrayType inputDiff,   \
      jint inputDiffOffset, long classPtr)                                  \
  {                                                                         \
    JNILRNUpdateGradInput<JArrayType, JType>(                               \
        env, thisClass, input, inputOffset, outputDiff, outputDiffOffset,   \
        inputDiff, inputDiffOffset, classPtr);                              \
  }

#ifdef __cplusplus
extern "C" {
#endif

// double
LRNInit(Double, jdouble, jdoubleArray);
LRNForward(Double, jdouble, jdoubleArray);
LRNBackward(Double, jdouble, jdoubleArray);

// float
LRNInit(Float, jfloat, jfloatArray);
LRNForward(Float, jfloat, jfloatArray);
LRNBackward(Float, jfloat, jfloatArray);

#ifdef __cplusplus
}
#endif
