#include <jni.h>

#include "debug.h"
#include "layer.h"
#include "memory.h"
#include "utils.h"

template <typename DType>
class MKLLinear : public MKLLayer<DType>
{
 public:
  MKLLinear();
  ~MKLLinear();

  void init(size_t inputHeight, size_t inputWidth, size_t outputChannel,
            size_t kernelHeight, size_t kernelWidth, const char *name);

  void updateOutput(DType *input, DType *output);
  void updateGradInput(DType *input, DType *gradOutput, DType *gradInput);
  void updateGradKernel(DType *input, DType *gradOutput, DType *gradKernel);
  void updateGradBias(DType *input, DType *gradOutput, DType *gradBias);

  std::shared_ptr<MKLData<DType>> kernel;
  std::shared_ptr<MKLData<DType>> bias;

  std::shared_ptr<MKLData<DType>> gradKernel;
  std::shared_ptr<MKLData<DType>> gradBias;

 private:
  // this method is not the same as createMklLayout in MKLMemory
  void firstPass();
  void preExecute(DType *input);

  size_t inputSize[2];
  size_t inputStrides[2];

  size_t outputSize[2];
  size_t outputStrides[2];

  size_t kernelSize[2];
  size_t kernelStrides[2];

  size_t biasSize[1];
  size_t biasStrides[1];

  size_t outputChannel;

  dnnPrimitive_t gradKernelPrim, gradBiasPrim;
};

template <typename DType>
MKLLinear<DType>::MKLLinear()
    : kernel(new MKLData<DType>),
      bias(new MKLData<DType>),
      gradKernel(new MKLData<DType>),
      gradBias(new MKLData<DType>),
      outputChannel(0),
      gradKernelPrim(NULL),
      gradBiasPrim(NULL)
{
}

template <typename DType>
MKLLinear<DType>::~MKLLinear()
{
  dnnDelete<DType>(gradKernelPrim);
  dnnDelete<DType>(gradBiasPrim);
}

template <typename DType>
void MKLLinear<DType>::init(size_t inputHeight, size_t inputWidth,
                            size_t outputChannel, size_t kernelHeight,
                            size_t kernelWidth, const char *name)
{
  this->dimension = 2;
  this->name.assign(name);

  inputSize[0] = inputWidth;
  inputSize[1] = inputHeight;

  outputSize[0] = outputChannel;
  outputSize[1] = inputHeight;

  kernelSize[0] = kernelWidth;
  kernelSize[1] = kernelHeight;

  inputStrides[0]  = 1;
  kernelStrides[0] = 1;
  outputStrides[0] = 1;
  for (int i = 1; i < this->dimension; i++) {
    inputStrides[i]  = inputStrides[i - 1] * inputSize[i - 1];
    kernelStrides[i] = kernelStrides[i - 1] * kernelSize[i - 1];
    outputStrides[i] = outputStrides[i - 1] * outputSize[i - 1];
  }

  biasSize[0]    = outputChannel;
  biasStrides[0] = 1;

  this->outputChannel = outputChannel;

  // create usr layout
  this->input->createUsrLayout(this->dimension, inputSize, inputStrides);
  this->output->createUsrLayout(this->dimension, outputSize, outputStrides);
  this->kernel->createUsrLayout(this->dimension, kernelSize, kernelStrides);
  this->bias->createUsrLayout(1, biasSize, biasStrides);

  this->gradInput->createUsrLayout(this->dimension, inputSize, inputStrides);
  this->gradOutput->createUsrLayout(this->dimension, outputSize, outputStrides);
  this->gradKernel->createUsrLayout(this->dimension, kernelSize, kernelStrides);
  // bias dimension is 1
  this->gradBias->createUsrLayout(1, biasSize, biasStrides);
}

template <typename DType>
void MKLLinear<DType>::firstPass()
{
  dnnError_t status = E_UNIMPLEMENTED;
  // forward
  status = dnnInnerProductCreateForwardBias<DType>(
      &(this->forwardPrim), NULL, this->dimension, inputSize, outputChannel);
  CHECK_EQ(status, E_SUCCESS);

  this->input->createMklLayout(this->forwardPrim, dnnResourceSrc);
  this->output->createMklLayout(this->forwardPrim, dnnResourceDst);
  this->kernel->createMklLayout(this->forwardPrim, dnnResourceFilter);
  this->bias->createMklLayout(this->forwardPrim, dnnResourceBias);

  // backward data
  status = dnnInnerProductCreateBackwardData<DType>(
      &(this->backwardPrim), NULL, this->dimension, inputSize, outputChannel);
  CHECK_EQ(status, E_SUCCESS);

  this->gradOutput->createMklLayout(this->backwardPrim, dnnResourceDiffDst);
  this->gradInput->createMklLayout(this->backwardPrim, dnnResourceDiffSrc);

  // backward kernel
  status = dnnInnerProductCreateBackwardFilter<DType>(
      &gradKernelPrim, NULL, this->dimension, inputSize, outputChannel);
  CHECK_EQ(status, E_SUCCESS);

  this->gradKernel->createMklLayout(this->gradKernelPrim,
                                    dnnResourceDiffFilter);

  // backward bias
  status = dnnInnerProductCreateBackwardBias<DType>(
      &gradBiasPrim, NULL, this->dimension, outputSize);
  CHECK_EQ(status, E_SUCCESS);

  this->gradBias->createMklLayout(this->gradBiasPrim, dnnResourceDiffBias);

  // we create the layout only at the first time
  this->isFirstPass = false;
}

template <typename DType>
void MKLLinear<DType>::preExecute(DType *input)
{
  caffe::cpu::OpenMpManager::setGpuDisabled();
  caffe::cpu::OpenMpManager::bindOpenMpThreads();

  this->input->createConversion();
  this->kernel->createConversion();
  this->bias->createConversion();
}

template <typename DType>
void MKLLinear<DType>::updateOutput(DType *input, DType *output)
{
  if (this->isFirstPass) firstPass();

  // Because the address will change every time, so we need create conversion
  // every forward/backward.
  // TODO Should we set the kernel and bias address every time?
  preExecute(input);
  this->output->createConversion();

#ifdef DEBUG
  printData<DType>(reinterpret_cast<DType *>(this->input->getUsrData()),
                   this->inputSize[3], this->inputSize[2], this->inputSize[1],
                   this->inputSize[0], "Forward input");
#endif

  dnnError_t status;
  void *resources[dnnResourceNumber];

  resources[dnnResourceFilter] = this->kernel->getConvertedData();
  resources[dnnResourceBias]   = this->bias->getConvertedData();
  resources[dnnResourceSrc]    = this->input->getConvertedData();
  resources[dnnResourceDst]    = this->output->getData();

  PERFSTART();
  status = dnnExecute<DType>(this->forwardPrim, resources);
  PERFEND("main computing");
  CHECK_EQ(status, E_SUCCESS);

  this->input->setIsConverted(true);
  this->kernel->setIsConverted(true);

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
void MKLLinear<DType>::updateGradInput(DType *input, DType *gradOutput,
                                       DType *gradInput)
{
  dnnError_t status;
  void *resources[dnnResourceNumber];

  preExecute(input);

  this->gradOutput->createConversion();
  this->gradInput->createConversion();

  resources[dnnResourceDiffDst] = this->gradOutput->getConvertedData();
  resources[dnnResourceFilter]  = this->kernel->getConvertedData();
  resources[dnnResourceDiffSrc] = this->gradInput->getData();

  // 4. main computing parts.
  PERFSTART();
  status = dnnExecute<DType>(this->backwardPrim, resources);
  CHECK_EQ(status, E_SUCCESS);
  PERFEND("main computing");

  this->gradOutput->setIsConverted(true);
  this->kernel->setIsConverted(false);

  if (!this->gradInput->isUsePrev()) {
    this->gradInput->backToUsr();
  }

#ifdef DEBUG
  printData<DType>(reinterpret_cast<DType *>(this->gradInput->getUsrData()),
                   inputSize[3], inputSize[2], inputSize[1], inputSize[0],
                   "backward gradient input");
#endif
}

template <typename DType>
void MKLLinear<DType>::updateGradKernel(DType *input, DType *gradOutput,
                                        DType *gradKernel)
{
  dnnError_t status;
  void *resources[dnnResourceNumber];

  preExecute(input);

  this->gradOutput->createConversion();
  this->gradKernel->createConversion();

  resources[dnnResourceDiffDst]    = this->gradOutput->getConvertedData();
  resources[dnnResourceSrc]        = this->input->getConvertedData();
  resources[dnnResourceDiffFilter] = this->gradKernel->getData();

  // 4. main computing parts.
  PERFSTART();
  status = dnnExecute<DType>(this->gradKernelPrim, resources);
  CHECK_EQ(status, E_SUCCESS);
  PERFEND("main computing");

  this->input->setIsConverted(false);

  // the kernel need not re-use for previous layer
  this->gradKernel->backToUsr();
}

template <typename DType>
void MKLLinear<DType>::updateGradBias(DType *input, DType *gradOutput,
                                      DType *gradBias)
{
  dnnError_t status;
  void *resources[dnnResourceNumber];

  preExecute(input);

  this->gradOutput->createConversion();
  this->gradBias->createConversion();

  resources[dnnResourceDiffDst]  = this->gradOutput->getConvertedData();
  resources[dnnResourceDiffBias] = this->gradBias->getData();

  // 4. main computing parts.
  PERFSTART();
  status = dnnExecute<DType>(this->gradBiasPrim, resources);
  CHECK_EQ(status, E_SUCCESS);
  PERFEND("main computing");

  this->gradOutput->setIsConverted(false);

  this->gradBias->backToUsr();
}

template <typename ArrayType, typename DType>
jlong JNILinearInit(JNIEnv *env, jclass thisClass, jint inputHeight,
                    jint inputWidth, jint outputChannel, jint kernelHeight,
                    jint kernelWidth, jstring name)
{
  const char *jName = env->GetStringUTFChars(name, NULL);
  MKLLinear<DType> *ptr = new MKLLinear<DType>();
  ptr->init(inputHeight, inputWidth, outputChannel, kernelHeight, kernelWidth,
            jName);

  return reinterpret_cast<long>(ptr);
}

template <typename ArrayType, typename DType>
void JNILinearUpdateOutput(JNIEnv *env, jclass thisClass, ArrayType input,
                           jint inputOffset, ArrayType output,
                           jint outputOffset, ArrayType kernel,
                           jint kernelOffset, ArrayType bias, jint biasOffset,
                           long classPtr)
{
  MKLLinear<DType> *ptr = reinterpret_cast<MKLLinear<DType> *>(classPtr);

  std::shared_ptr<ZipArray<ArrayType, DType>> jInput(
      new ZipArray<ArrayType, DType>(env, input, inputOffset, ptr->input));

  std::shared_ptr<ZipArray<ArrayType, DType>> jOutput(
      new ZipArray<ArrayType, DType>(env, output, outputOffset, ptr->output));

  std::shared_ptr<ZipArray<ArrayType, DType>> jKernel(
      new ZipArray<ArrayType, DType>(env, kernel, kernelOffset, ptr->kernel));

  std::shared_ptr<ZipArray<ArrayType, DType>> jBias(
      new ZipArray<ArrayType, DType>(env, bias, biasOffset, ptr->bias));

  ptr->updateOutput(jInput->getPtr(), jOutput->getPtr());
}

template <typename ArrayType, typename DType>
void JNILinearUpdateGradInput(JNIEnv *env, jclass thisClass, ArrayType input,
                              jint inputOffset, ArrayType outputDiff,
                              jint outputDiffOffset, ArrayType inputDiff,
                              jint inputDiffOffset, ArrayType kernel,
                              jint kernelOffset, ArrayType bias,
                              jint biasOffset, long classPtr)
{
  MKLLinear<DType> *ptr = reinterpret_cast<MKLLinear<DType> *>(classPtr);
  std::shared_ptr<ZipArray<ArrayType, DType>> jInput(
      new ZipArray<ArrayType, DType>(env, input, inputOffset, ptr->input));

  std::shared_ptr<ZipArray<ArrayType, DType>> jOutputDiff(
      new ZipArray<ArrayType, DType>(env, outputDiff, outputDiffOffset,
                                     ptr->gradOutput));

  std::shared_ptr<ZipArray<ArrayType, DType>> jInputDiff(
      new ZipArray<ArrayType, DType>(env, inputDiff, inputDiffOffset,
                                     ptr->gradInput));

  std::shared_ptr<ZipArray<ArrayType, DType>> jKernel(
      new ZipArray<ArrayType, DType>(env, kernel, kernelOffset, ptr->kernel));

  std::shared_ptr<ZipArray<ArrayType, DType>> jBias(
      new ZipArray<ArrayType, DType>(env, bias, biasOffset, ptr->bias));

  ptr->updateGradInput(jInput->getPtr(), jOutputDiff->getPtr(),
                       jInputDiff->getPtr());
}

template <typename ArrayType, typename DType>
void JNILinearUpdateGradKernel(JNIEnv *env, jclass thisClass, ArrayType input,
                               jint inputOffset, ArrayType outputDiff,
                               jint outputDiffOffset, ArrayType kernelDiff,
                               jint kernelDiffOffset, ArrayType kernel,
                               jint kernelOffset, ArrayType bias,
                               jint biasOffset, long classPtr)
{
  MKLLinear<DType> *ptr = reinterpret_cast<MKLLinear<DType> *>(classPtr);

  std::shared_ptr<ZipArray<ArrayType, DType>> jInput(
      new ZipArray<ArrayType, DType>(env, input, inputOffset, ptr->input));

  std::shared_ptr<ZipArray<ArrayType, DType>> jOutputDiff(
      new ZipArray<ArrayType, DType>(env, outputDiff, outputDiffOffset,
                                     ptr->gradOutput));

  std::shared_ptr<ZipArray<ArrayType, DType>> jKernelDiff(
      new ZipArray<ArrayType, DType>(env, kernelDiff, kernelDiffOffset,
                                     ptr->gradKernel));

  std::shared_ptr<ZipArray<ArrayType, DType>> jKernel(
      new ZipArray<ArrayType, DType>(env, kernel, kernelOffset, ptr->kernel));

  std::shared_ptr<ZipArray<ArrayType, DType>> jBias(
      new ZipArray<ArrayType, DType>(env, bias, biasOffset, ptr->bias));

  ptr->updateGradKernel(jInput->getPtr(), jOutputDiff->getPtr(),
                        jKernelDiff->getPtr());
}

template <typename ArrayType, typename DType>
void JNILinearUpdateGradBias(JNIEnv *env, jclass thisClass, ArrayType input,
                             jint inputOffset, ArrayType outputDiff,
                             jint outputDiffOffset, ArrayType biasDiff,
                             jint biasDiffOffset, ArrayType kernel,
                             jint kernelOffset, ArrayType bias, jint biasOffset,
                             long classPtr)
{
  MKLLinear<DType> *ptr = reinterpret_cast<MKLLinear<DType> *>(classPtr);

  std::shared_ptr<ZipArray<ArrayType, DType>> jInput(
      new ZipArray<ArrayType, DType>(env, input, inputOffset, ptr->input));

  std::shared_ptr<ZipArray<ArrayType, DType>> jOutputDiff(
      new ZipArray<ArrayType, DType>(env, outputDiff, outputDiffOffset,
                                     ptr->gradOutput));

  std::shared_ptr<ZipArray<ArrayType, DType>> jBiasDiff(
      new ZipArray<ArrayType, DType>(env, biasDiff, biasDiffOffset,
                                     ptr->gradBias));

  std::shared_ptr<ZipArray<ArrayType, DType>> jKernel(
      new ZipArray<ArrayType, DType>(env, kernel, kernelOffset, ptr->kernel));

  std::shared_ptr<ZipArray<ArrayType, DType>> jBias(
      new ZipArray<ArrayType, DType>(env, bias, biasOffset, ptr->bias));

  ptr->updateGradBias(jInput->getPtr(), jOutputDiff->getPtr(),
                      jBiasDiff->getPtr());
}
// Macro
#define LinearInit(DType, JType, JArrayType)                                \
  JNIEXPORT                                                                 \
  jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_LinearInit##DType( \
      JNIEnv *env, jclass thisClass, jint inputHeight, jint inputWidth,     \
      jint outputChannel, jint kernelHeight, jint kernelWidth, jstring name)              \
  {                                                                         \
    return JNILinearInit<JArrayType, JType>(env, thisClass, inputHeight,    \
                                            inputWidth, outputChannel,      \
                                            kernelHeight, kernelWidth, name);     \
  }

#define LinearForward(DType, JType, JArrayType)                               \
  JNIEXPORT                                                                   \
  void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_LinearForward##DType( \
      JNIEnv *env, jclass thisClass, JArrayType input, jint inputOffset,      \
      JArrayType output, jint outputOffset, JArrayType kernel,                \
      jint kernelOffset, JArrayType bias, jint biasOffset, long classPtr)     \
  {                                                                           \
    JNILinearUpdateOutput<JArrayType, JType>(                                 \
        env, thisClass, input, inputOffset, output, outputOffset, kernel,     \
        kernelOffset, bias, biasOffset, classPtr);                            \
  }

#define LinearBackwardData(DType, JType, JArrayType)                          \
  JNIEXPORT                                                                   \
  void JNICALL                                                                \
      Java_com_intel_analytics_bigdl_mkl_MKL_LinearBackwardData##DType(     \
          JNIEnv *env, jclass thisClass, JArrayType input, jint inputOffset,  \
          JArrayType outputDiff, jint outputDiffOffset, JArrayType inputDiff, \
          jint inputDiffOffset, JArrayType kernel, jint kernelOffset,         \
          JArrayType bias, jint biasOffset, long classPtr)                    \
  {                                                                           \
    JNILinearUpdateGradInput<JArrayType, JType>(                              \
        env, thisClass, input, inputOffset, outputDiff, outputDiffOffset,     \
        inputDiff, inputDiffOffset, kernel, kernelOffset, bias, biasOffset,   \
        classPtr);                                                            \
  }

#define LinearBackwardKernel(DType, JType, JArrayType)                         \
  JNIEXPORT                                                                    \
  void JNICALL                                                                 \
      Java_com_intel_analytics_bigdl_mkl_MKL_LinearBackwardKernel##DType(    \
          JNIEnv *env, jclass thisClass, JArrayType input, jint inputOffset,   \
          JArrayType outputDiff, jint outputDiffOffset, JArrayType kernelDiff, \
          jint kernelDiffOffset, JArrayType kernel, jint kernelOffset,         \
          JArrayType bias, jint biasOffset, long classPtr)                     \
  {                                                                            \
    JNILinearUpdateGradKernel<JArrayType, JType>(                              \
        env, thisClass, input, inputOffset, outputDiff, outputDiffOffset,      \
        kernelDiff, kernelDiffOffset, kernel, kernelOffset, bias, biasOffset,  \
        classPtr);                                                             \
  }

#define LinearBackwardBias(DType, JType, JArrayType)                         \
  JNIEXPORT                                                                  \
  void JNICALL                                                               \
      Java_com_intel_analytics_bigdl_mkl_MKL_LinearBackwardBias##DType(    \
          JNIEnv *env, jclass thisClass, JArrayType input, jint inputOffset, \
          JArrayType outputDiff, jint outputDiffOffset, JArrayType biasDiff, \
          jint biasDiffOffset, JArrayType kernel, jint kernelOffset,         \
          JArrayType bias, jint biasOffset, long classPtr)                   \
  {                                                                          \
    JNILinearUpdateGradBias<JArrayType, JType>(                              \
        env, thisClass, input, inputOffset, outputDiff, outputDiffOffset,    \
        biasDiff, biasDiffOffset, kernel, kernelOffset, bias, biasOffset,    \
        classPtr);                                                           \
  }

#ifdef __cplusplus
extern "C" {
#endif

// double
LinearInit(Double, jdouble, jdoubleArray);
LinearForward(Double, jdouble, jdoubleArray);
LinearBackwardData(Double, jdouble, jdoubleArray);
LinearBackwardKernel(Double, jdouble, jdoubleArray);
LinearBackwardBias(Double, jdouble, jdoubleArray);

// float
LinearInit(Float, jfloat, jfloatArray);
LinearForward(Float, jfloat, jfloatArray);
LinearBackwardData(Float, jfloat, jfloatArray);
LinearBackwardKernel(Float, jfloat, jfloatArray);
LinearBackwardBias(Float, jfloat, jfloatArray);

#ifdef __cplusplus
}
#endif
