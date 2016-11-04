#include <jni.h>

#include "debug.h"
#include "layer.h"
#include "memory.h"
#include "utils.h"

template <typename DType>
class MKLBatchNorm : public MKLLayer<DType>
{
 public:
  MKLBatchNorm();
  ~MKLBatchNorm();

  void init(size_t inputNumber, size_t inputChannel, size_t inputHeight,
            size_t inputWidth, DType eps, int useKernel, int useBias,
            int dimension, const char *name);

  void updateOutput(DType *input, DType *output);
  void updateGradInput(DType *input, DType *gradOutput, DType *gradInput);

  void setKernel(DType *ptr);
  void setBias(DType *ptr);
  void setGradKernel(DType *ptr);
  void setGradBias(DType *ptr);

 private:
  // this method is not the same as createMklLayout in MKLMemory
  void firstPass();
  void preExecute(DType *input);

  std::shared_ptr<MKLData<DType>> scaleShift;
  std::shared_ptr<MKLData<DType>> workspace;

  size_t inputSize[4];
  size_t inputStrides[4];

  size_t outputSize[4];
  size_t outputStrides[4];

  DType eps;
  bool useKernel;
  bool useBias;

  DType *kernel;
  DType *bias;
  DType *gradKernel;
  DType *gradBias;

  dnnPrimitive_t scaleShiftPrim;
};

template <typename DType>
MKLBatchNorm<DType>::MKLBatchNorm()
    : scaleShift(new MKLData<DType>),
      workspace(new MKLData<DType>),
      kernel(NULL),
      bias(NULL),
      gradKernel(NULL),
      gradBias(NULL),
      scaleShiftPrim(NULL),
      useKernel(true),
      useBias(true)
{
  eps = 0.00001;
}

template <typename DType>
MKLBatchNorm<DType>::~MKLBatchNorm()
{
  dnnDelete<DType>(scaleShiftPrim);
}

template <typename DType>
void MKLBatchNorm<DType>::setKernel(DType *ptr)
{
  kernel = ptr;
}
template <typename DType>
void MKLBatchNorm<DType>::setBias(DType *ptr)
{
  bias = ptr;
}
template <typename DType>
void MKLBatchNorm<DType>::setGradKernel(DType *ptr)
{
  gradKernel = ptr;
}
template <typename DType>
void MKLBatchNorm<DType>::setGradBias(DType *ptr)
{
  gradBias = ptr;
}

template <typename DType>
void MKLBatchNorm<DType>::init(size_t inputNumber, size_t inputChannel,
                               size_t inputHeight, size_t inputWidth,
                               DType eps, int useKernel, int useBias,
                               int dimension, const char *name)
{
  this->dimension = dimension;
  this->name.assign(name);

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

  this->eps       = eps;
  this->useKernel = useKernel > 0 ? true : false;
  this->useBias   = useBias > 0 ? true : false;

  // create usr layout
  this->input->createUsrLayout(dimension, inputSize, inputStrides);
  this->output->createUsrLayout(dimension, outputSize, outputStrides);

  this->gradInput->createUsrLayout(dimension, inputSize, inputStrides);
  this->gradOutput->createUsrLayout(dimension, outputSize, outputStrides);
}

template <typename DType>
void MKLBatchNorm<DType>::firstPass()
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

  // forward
  status = dnnBatchNormalizationCreateForward<DType>(&(this->forwardPrim), NULL,
                                                     layout, eps);
  CHECK_EQ(status, E_SUCCESS);

  this->input->createMklLayout(this->forwardPrim, dnnResourceSrc);
  this->output->createMklLayout(this->forwardPrim, dnnResourceDst);

  // backward data
  status = dnnBatchNormalizationCreateBackwardData<DType>(&(this->backwardPrim),
                                                          NULL, layout, eps);
  CHECK_EQ(status, E_SUCCESS);

  this->gradOutput->createMklLayout(this->backwardPrim, dnnResourceDiffDst);
  this->gradInput->createMklLayout(this->backwardPrim, dnnResourceDiffSrc);

  // scaleshift
  this->scaleShift->createMklLayout(this->forwardPrim, dnnResourceScaleShift);
  this->scaleShift->createConversion(true);
  if (useKernel) {
    status = dnnBatchNormalizationCreateBackwardScaleShift<DType>(
        &scaleShiftPrim, NULL, layout, eps);
    CHECK_EQ(status, E_SUCCESS);
  }

  // workspace
  this->workspace->createMklLayout(this->forwardPrim, dnnResourceWorkspace);
  this->workspace->createConversion(true);

  // we create the layout only at the first time
  this->isFirstPass = false;

  // delte the layout
  if (!this->input->isUsePrev()) {
    dnnLayoutDelete<DType>(layout);
  }
}

template <typename DType>
void MKLBatchNorm<DType>::preExecute(DType *input)
{
  if (this->isUseOpenMpManager) {
    caffe::cpu::OpenMpManager::setGpuDisabled();
    caffe::cpu::OpenMpManager::bindOpenMpThreads();
  }

  this->input->createConversion();
}

template <typename DType>
void MKLBatchNorm<DType>::updateOutput(DType *input, DType *output)
{
  if (this->isFirstPass) firstPass();

  // Because the address will change every time, so we need create conversion
  // every forward/backward.
  // TODO Should we set the kernel and bias address every time?
  preExecute(input);
  this->output->createConversion();

  // workspace->setZero();
  // scaleShift->setZero();

  DType *ptr = reinterpret_cast<DType *>(scaleShift->getData());

  // pad the scale shift with kernel and bias
  if (useKernel) {
    for (int i = 0; i < inputSize[2]; i++) {
      ptr[i] = kernel[i];
      if (useBias)
        ptr[i + inputSize[2]] = bias[i];
      else
        ptr[i + inputSize[2]] = 0;
    }
  } else {
    for (int i = 0; i < inputSize[2]; i++) {
      ptr[i]                = 1.0;
      ptr[i + inputSize[2]] = 0;
    }
  }

#ifdef DEBUG
  printData<DType>(reinterpret_cast<DType *>(this->input->getUsrData()),
                   this->inputSize[3], this->inputSize[2], this->inputSize[1],
                   this->inputSize[0], "Forward input");
#endif

  dnnError_t status;
  void *resources[dnnResourceNumber];

  resources[dnnResourceSrc]        = this->input->getConvertedData();
  resources[dnnResourceDst]        = this->output->getData();
  resources[dnnResourceScaleShift] = scaleShift->getData();
  resources[dnnResourceWorkspace]  = workspace->getData();

  PERFSTART();
  status = dnnExecute<DType>(this->forwardPrim, resources);
  PERFEND("main computing");
  CHECK_EQ(status, E_SUCCESS);

  this->input->setIsConverted(true);

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
void MKLBatchNorm<DType>::updateGradInput(DType *input, DType *gradOutput,
                                          DType *gradInput)
{
  dnnError_t status;
  void *resources[dnnResourceNumber];

  preExecute(input);

  this->gradOutput->createConversion();
  this->gradInput->createConversion();

  resources[dnnResourceDiffDst]    = this->gradOutput->getConvertedData();
  resources[dnnResourceDiffSrc]    = this->gradInput->getData();
  resources[dnnResourceSrc]        = this->input->getConvertedData();
  resources[dnnResourceScaleShift] = scaleShift->getData();
  resources[dnnResourceWorkspace]  = workspace->getData();

  // 4. main computing parts.
  PERFSTART();
  status = dnnExecute<DType>(this->backwardPrim, resources);
  CHECK_EQ(status, E_SUCCESS);
  PERFEND("main computing");

  this->input->setIsConverted(false);

  if (useKernel) {
    void *diffRes[dnnResourceNumber];
    diffRes[dnnResourceDiffDst]        = this->gradOutput->getConvertedData();
    diffRes[dnnResourceSrc]            = this->input->getConvertedData();
    diffRes[dnnResourceDiffScaleShift] = scaleShift->getData();
    diffRes[dnnResourceWorkspace]      = workspace->getData();

    PERFSTART();
    status = dnnExecute<DType>(scaleShiftPrim, diffRes);
    CHECK_EQ(status, E_SUCCESS);
    PERFEND("weight and bias diff main computing");

    DType *ptr = reinterpret_cast<DType *>(scaleShift->getData());
    for (int i = 0; i < inputSize[2]; i++) {
      gradKernel[i] = ptr[i];
      gradBias[i] = 0;
      if (useBias) {
        gradBias[i] = ptr[i + inputSize[2]];
      }
    }
  }

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
jlong JNIBatchNormInit(JNIEnv *env, jclass thisClass, jint inputNumber,
                       jint inputChannel, jint inputHeight, jint inputWidth,
                       DType eps, jint useKernel, jint useBias, jint dimension,
                       jstring name)
{
  const char *jName = env->GetStringUTFChars(name, NULL);
  MKLBatchNorm<DType> *ptr = new MKLBatchNorm<DType>();
  ptr->init(inputNumber, inputChannel, inputHeight, inputWidth, eps, useKernel,
            useBias, dimension, jName);

  return reinterpret_cast<long>(ptr);
}

template <typename ArrayType, typename DType>
void JNIBatchNormUpdateOutput(JNIEnv *env, jclass thisClass, ArrayType input,
                              jint inputOffset, ArrayType output,
                              jint outputOffset, ArrayType kernel,
                              jint kernelOffset, ArrayType bias,
                              jint biasOffset, long classPtr)
{
  MKLBatchNorm<DType> *ptr = reinterpret_cast<MKLBatchNorm<DType> *>(classPtr);

  std::shared_ptr<ZipArray<ArrayType, DType>> jInput(
      new ZipArray<ArrayType, DType>(env, input, inputOffset, ptr->input));

  std::shared_ptr<ZipArray<ArrayType, DType>> jOutput(
      new ZipArray<ArrayType, DType>(env, output, outputOffset, ptr->output));

  std::shared_ptr<ZipArray<ArrayType, DType>> jKernel(
      new ZipArray<ArrayType, DType>(env, kernel, kernelOffset, NULL));

  std::shared_ptr<ZipArray<ArrayType, DType>> jBias(
      new ZipArray<ArrayType, DType>(env, bias, biasOffset, NULL));

  ptr->setKernel(jKernel->getPtr());
  ptr->setBias(jBias->getPtr());

  ptr->updateOutput(jInput->getPtr(), jOutput->getPtr());
}

template <typename ArrayType, typename DType>
void JNIBatchNormUpdateGradInput(JNIEnv *env, jclass thisClass, ArrayType input,
                                 jint inputOffset, ArrayType outputDiff,
                                 jint outputDiffOffset, ArrayType inputDiff,
                                 jint inputDiffOffset, ArrayType kernelDiff,
                                 jint kernelDiffOffset, ArrayType biasDiff,
                                 jint biasDiffOffset, long classPtr)
{
  MKLBatchNorm<DType> *ptr = reinterpret_cast<MKLBatchNorm<DType> *>(classPtr);
  std::shared_ptr<ZipArray<ArrayType, DType>> jInput(
      new ZipArray<ArrayType, DType>(env, input, inputOffset, ptr->input));

  std::shared_ptr<ZipArray<ArrayType, DType>> jOutputDiff(
      new ZipArray<ArrayType, DType>(env, outputDiff, outputDiffOffset,
                                     ptr->gradOutput));

  std::shared_ptr<ZipArray<ArrayType, DType>> jInputDiff(
      new ZipArray<ArrayType, DType>(env, inputDiff, inputDiffOffset,
                                     ptr->gradInput));

  std::shared_ptr<ZipArray<ArrayType, DType>> jKernelDiff(
      new ZipArray<ArrayType, DType>(env, kernelDiff, kernelDiffOffset, NULL));

  std::shared_ptr<ZipArray<ArrayType, DType>> jBiasDiff(
      new ZipArray<ArrayType, DType>(env, biasDiff, biasDiffOffset, NULL));

  ptr->setGradKernel(jKernelDiff->getPtr());
  ptr->setGradBias(jBiasDiff->getPtr());

  ptr->updateGradInput(jInput->getPtr(), jOutputDiff->getPtr(),
                       jInputDiff->getPtr());
}

// Macro
#define BatchNormInit(DType, JType, JArrayType)                                \
  JNIEXPORT                                                                    \
  jlong JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_BatchNormInit##DType( \
      JNIEnv *env, jclass thisClass, jint inputNumber, jint inputChannel,      \
      jint inputHeight, jint inputWidth, JType eps, jint useKernel,          \
      jint useBias, jint dimension, jstring name)                                            \
  {                                                                            \
    return JNIBatchNormInit<JArrayType, JType>(                                \
        env, thisClass, inputNumber, inputChannel, inputHeight, inputWidth,    \
        eps, useKernel, useBias, dimension, name);                                   \
  }

#define BatchNormForward(DType, JType, JArrayType)                            \
  JNIEXPORT                                                                   \
  void JNICALL                                                                \
      Java_com_intel_analytics_sparkdl_mkl_MKL_BatchNormForward##DType(       \
          JNIEnv *env, jclass thisClass, JArrayType input, jint inputOffset,  \
          JArrayType output, jint outputOffset, JArrayType kernel,            \
          jint kernelOffset, JArrayType bias, jint biasOffset, long classPtr) \
  {                                                                           \
    JNIBatchNormUpdateOutput<JArrayType, JType>(                              \
        env, thisClass, input, inputOffset, output, outputOffset, kernel,     \
        kernelOffset, bias, biasOffset, classPtr);                            \
  }

#define BatchNormBackward(DType, JType, JArrayType)                           \
  JNIEXPORT                                                                   \
  void JNICALL                                                                \
      Java_com_intel_analytics_sparkdl_mkl_MKL_BatchNormBackward##DType(      \
          JNIEnv *env, jclass thisClass, JArrayType input, jint inputOffset,  \
          JArrayType outputDiff, jint outputDiffOffset, JArrayType inputDiff, \
          jint inputDiffOffset, JArrayType kernelDiff, jint kernelDiffOffset, \
          JArrayType biasDiff, jint biasDiffOffset, long classPtr)            \
  {                                                                           \
    JNIBatchNormUpdateGradInput<JArrayType, JType>(                           \
        env, thisClass, input, inputOffset, outputDiff, outputDiffOffset,     \
        inputDiff, inputDiffOffset, kernelDiff, kernelDiffOffset, biasDiff,   \
        biasDiffOffset, classPtr);                                            \
  }

#ifdef __cplusplus
extern "C" {
#endif

// double
BatchNormInit(Double, jdouble, jdoubleArray);
BatchNormForward(Double, jdouble, jdoubleArray);
BatchNormBackward(Double, jdouble, jdoubleArray);

// float
BatchNormInit(Float, jfloat, jfloatArray);
BatchNormForward(Float, jfloat, jfloatArray);
BatchNormBackward(Float, jfloat, jfloatArray);

#ifdef __cplusplus
}
#endif
