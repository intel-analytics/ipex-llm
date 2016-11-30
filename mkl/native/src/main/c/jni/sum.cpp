#include <stdio.h>
#include <vector>
#include <cstring>

#include "debug.h"
#include "layer.h"
#include "memory.h"
#include "utils.h"

using namespace std;

template <typename DType>
class MKLSum : public MKLLayer<DType>
{
 public:
  MKLSum();
  ~MKLSum();

  void init(int numSums, int dimension, int *size);
  void setIPrev(int index, long curr);

  void updateOutput(DType *input, DType **output);
  void updateGradInput(DType *gradInput, DType **gradOutput);

  // attention, we will override the four variables of MKLLayer
  vector<shared_ptr<MKLData<DType>>> gradOutput;
  vector<shared_ptr<MKLData<DType>>> output;

 private:
  void firstPass();
  void preExecute(DType *input);

  int numSums;  // number of concats
  DType *coefficients;
};

template <typename DType>
MKLSum<DType>::MKLSum() : numSums(0)
{
  // TODO
}

template <typename DType>
MKLSum<DType>::~MKLSum()
{
  // TODO
  delete[] coefficients;
}

template <typename DType>
void MKLSum<DType>::setIPrev(int index, long curr)
{
  MKLLayer<DType> *ptr = reinterpret_cast<MKLLayer<DType> *>(curr);
  if (index < this->gradOutput.size()) {
    this->output[index]->setMklData(this->input->getData(),
                                    this->input->getUsrData() !=
                                    this->input->getMklData());

    ptr->input->setMklData(this->output[index]->getData(),
                           this->output[index]->getUsrData() !=
                           this->output[index]->getMklData());
    ptr->input->setUsePrev(true);
    this->output[index]->setUseNext(true);
    // LOG(DBG) << "output[" << index << "] = " << this->output[index]->isUseNext();

    this->gradOutput[index]->setMklData(ptr->gradInput->getData(),
                                        ptr->gradInput->getUsrData() !=
                                        ptr->gradInput->getMklData());
    this->gradOutput[index]->setUseNext(true);
    ptr->gradInput->setUsePrev(true);
    // LOG(DBG) << "OMIT CONVERSION";
  }
}

template <typename DType>
void MKLSum<DType>::init(int numSums, int dimension, int *size)
{
  this->numSums      = numSums;
  this->dimension    = dimension;
  this->coefficients = new DType[numSums];

  // LOG(DBG) << numSums;

  size_t inputSize[dimension];
  size_t inputStrides[dimension];
  //size_t outputSize[dimension];
  //size_t outputStrides[dimension];
  
  inputSize[0] = size[0];
  inputStrides[0] = 1;
  for (int i = 1; i < dimension; i++) {
    inputSize[i] = size[i];
    inputStrides[i] = inputSize[i-1] * inputStrides[i-1];
  }

  // for (int i = 0; i < dimension; i++) {
  //   LOG(DBG) << inputSize[i];
  //   LOG(DBG) << inputStrides[i];
  // }

  for (int i = 0; i < numSums; i++) {
    gradOutput.push_back(shared_ptr<MKLData<DType>>(new MKLData<DType>));
    output.push_back(shared_ptr<MKLData<DType>>(new MKLData<DType>));

    // set the size.
    // the size of every channel should be gaved in size.
    // the dimension of every channel should be the same.
    // inputStrides[0] = 1;
    // inputSize[0]    = size[offset];
    // for (int j = 1; j < dimension; j++) {
    //   inputSize[j]    = size[offset + j];
    //   inputStrides[j] = inputStrides[j - 1] * inputSize[j - 1];
    // }
    // offset += dimension;

    this->gradOutput[i]->createUsrLayout(dimension, inputSize, inputStrides);
    this->output[i]->createUsrLayout(dimension, inputSize, inputStrides);
    this->coefficients[i] = 1;  // TODO coefficients may be not 1.0
  }

  // TODO check size of all input, they should be the same

  this->input->createUsrLayout(dimension, inputSize, inputStrides);
  this->gradInput->createUsrLayout(dimension, inputSize, inputStrides);
}

template <typename DType>
void MKLSum<DType>::firstPass()
{
  dnnLayout_t layout = NULL;
  if (this->input->isUsePrev()) {
    layout = this->input->layoutPrev;
  }

  if (!layout) {
    layout  = this->input->getUsrLayout();
  }

  dnnError_t status = E_UNIMPLEMENTED;
  status = dnnSumCreate<DType>(&(this->backwardPrim), NULL, numSums, layout,
                               &this->coefficients[0]);
  CHECK_EQ(status, E_SUCCESS);

  this->input->createMklLayout(this->backwardPrim, dnnResourceDst);
  this->gradInput->createMklLayout(this->backwardPrim, dnnResourceDst);

  for (int i = 0; i < numSums; i++) {
    this->output[i]->createMklLayout(
        this->backwardPrim, (dnnResourceType_t)(dnnResourceMultipleSrc + i));
    this->gradOutput[i]->createMklLayout(
      this->backwardPrim, (dnnResourceType_t)(dnnResourceMultipleSrc + i));
  }

  this->isFirstPass = false;
}

template <typename DType>
void MKLSum<DType>::updateOutput(DType *input, DType **output)
{
  caffe::cpu::OpenMpManager::setGpuDisabled();
  caffe::cpu::OpenMpManager::bindOpenMpThreads();

  if (this->isFirstPass) firstPass();

  for (int i = 0; i < numSums; i++) {
    this->output[i]->setUsrData(output[i]);
    this->output[i]->createConversion();
  }
  this->input->setUsrData(input);
  this->input->createConversion();

  PERFSTART();
  for (int i = 0; i < numSums; i++) {
    // LOG(DBG) << "output[" << i << "] = " << this->output[i]->isUseNext();
    if (!this->output[i]->isUseNext()) {
      memcpy(this->output[i]->getData(), this->input->getConvertedData(),
             this->output[i]->getMklLayoutSize());
      // LOG(DBG) << "HELLO SUM COPY";
    }
  }
  PERFEND("sum copy");

  for (int i = 0; i < numSums; i++) {
    if (!this->output[i]->isUseNext())
      this->output[i]->backToUsr();
  }
}

template <typename DType>
void MKLSum<DType>::updateGradInput(DType *gradInput, DType **gradOutput)
{
  caffe::cpu::OpenMpManager::setGpuDisabled();
  caffe::cpu::OpenMpManager::bindOpenMpThreads();

  // Because the forward of sum will not be called.
  if (this->isFirstPass) firstPass();

  for (int i = 0; i < numSums; i++) {
    this->gradOutput[i]->setUsrData(gradOutput[i]);
    this->gradOutput[i]->createConversion();
  }
  this->gradInput->setUsrData(gradInput);
  this->gradInput->createConversion();

  dnnError_t status;
  void *resources[dnnResourceNumber];

  PERFSTART()
  for (int i = 0; i < numSums; i++) {
    resources[dnnResourceMultipleSrc + i] =
        this->gradOutput[i]->getConvertedData();
  }
  PERFEND("prepare gradOutput");
  resources[dnnResourceDst] = this->gradInput->getData();

  PERFSTART();
  status = dnnExecute<DType>(this->backwardPrim, resources);
  PERFEND("main computing");

  if (!this->gradInput->isUsePrev()) {
    this->gradInput->backToUsr();
  }
}

template <typename ArrayType, typename DType>
jlong JNISumInit(JNIEnv *env, jclass thisClass, int numSums, int dimension,
                 jintArray size)
{
  MKLSum<DType> *ptr = new MKLSum<DType>();

  jint *jSize =
      reinterpret_cast<int *>(env->GetPrimitiveArrayCritical(size, 0));
  ptr->init(numSums, dimension, jSize);
  env->ReleasePrimitiveArrayCritical(size, jSize, 0);

  return reinterpret_cast<long>(ptr);
}

template <typename ArrayType, typename DType>
void JNISumUpdateOutput(JNIEnv *env, jclass thisClass, ArrayType input,
                        jint inputOffset, jobjectArray output,
                        jintArray outputOffset, long classPtr)
{
  MKLSum<DType> *ptr = reinterpret_cast<MKLSum<DType> *>(classPtr);

  jint *jOutputOffset =
      reinterpret_cast<jint *>(env->GetPrimitiveArrayCritical(outputOffset, 0));

  // TODO we should re-write, this version makes a little complict.
  int len = env->GetArrayLength(output);
  DType *outputArrStart[len];
  DType *outputArr[len];
  ArrayType jOutputArr[len];
  for (int i = 0; i < len; i++) {
    jOutputArr[i]     = (ArrayType)(env->GetObjectArrayElement(output, i));
    outputArrStart[i] = reinterpret_cast<DType *>(
        env->GetPrimitiveArrayCritical(jOutputArr[i], 0));
    outputArr[i] = outputArrStart[i] + jOutputOffset[i];
  }

  std::shared_ptr<ZipArray<ArrayType, DType>> jInput(
      new ZipArray<ArrayType, DType>(env, input, inputOffset, ptr->input));

  ptr->updateOutput(jInput->getPtr(), outputArr);

  for (int i = 0; i < len; i++) {
    env->ReleasePrimitiveArrayCritical(jOutputArr[i], outputArrStart[i], 0);
  }

  env->ReleasePrimitiveArrayCritical(outputOffset, jOutputOffset, 0);
}

template <typename ArrayType, typename DType>
void JNISumUpdateGradInput(JNIEnv *env, jclass thisClass, ArrayType inputDiff,
                           jint inputDiffOffset, jobjectArray outputDiff,
                           jintArray outputDiffOffset, long classPtr)
{
  MKLSum<DType> *ptr = reinterpret_cast<MKLSum<DType> *>(classPtr);

  jint *jOutputDiffOffset = reinterpret_cast<jint *>(
      env->GetPrimitiveArrayCritical(outputDiffOffset, 0));

  // TODO we should re-write, this version makes a little complict.
  int len = env->GetArrayLength(outputDiff);
  DType *outputDiffArrStart[len];
  DType *outputDiffArr[len];
  ArrayType jOutputDiffArr[len];
  for (int i = 0; i < len; i++) {
    jOutputDiffArr[i] = (ArrayType)(env->GetObjectArrayElement(outputDiff, i));
    outputDiffArrStart[i] = reinterpret_cast<DType *>(
        env->GetPrimitiveArrayCritical(jOutputDiffArr[i], 0));
    outputDiffArr[i] = outputDiffArrStart[i] + jOutputDiffOffset[i];
  }

  std::shared_ptr<ZipArray<ArrayType, DType>> jInputDiff(
      new ZipArray<ArrayType, DType>(env, inputDiff, inputDiffOffset,
                                     ptr->gradInput));

  ptr->updateGradInput(jInputDiff->getPtr(), outputDiffArr);

  for (int i = 0; i < len; i++) {
    env->ReleasePrimitiveArrayCritical(jOutputDiffArr[i], outputDiffArrStart[i],
                                       0);
  }

  env->ReleasePrimitiveArrayCritical(outputDiffOffset, jOutputDiffOffset, 0);
}

template <typename ArrayType, typename DType>
void JNISumSetNext(JNIEnv *env, jclass thisClass, long next, int index,
                      long curr)
{
  MKLLayer<DType> *nextLayer = reinterpret_cast<MKLLayer<DType>*>(next);
  MKLSum<DType> *currLayer = reinterpret_cast<MKLSum<DType>*>(curr);

  if (nextLayer && currLayer && index < currLayer->gradOutput.size()) {
    if (nextLayer->gradInput->getMklLayout() &&
        nextLayer->gradInput->getMklData()) {
      currLayer->gradOutput[index]->layoutNext = nextLayer->gradInput->getMklLayout();
      currLayer->gradOutput[index]->dataNext = nextLayer->gradInput->getMklData();

      if (currLayer->gradOutput[index]->getMklData()) {
        dnnReleaseBuffer<DType>(currLayer->gradOutput[index]->getMklData());
        currLayer->gradOutput[index]->setMklData(NULL);
      }

      nextLayer->gradInput->setUsePrev(true);
      currLayer->gradOutput[index]->setUseNext(true);
    }
  }
}

// Macro
#define SumInit(DType, JType, JArrayType)                                    \
  JNIEXPORT                                                                  \
  jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_SumInit##DType(     \
      JNIEnv *env, jclass thisClass, jint numSums, jint dimension,           \
      jintArray size)                                                        \
  {                                                                          \
    return JNISumInit<JArrayType, JType>(env, thisClass, numSums, dimension, \
                                         size);                              \
  }

#define SumForward(DType, JType, JArrayType)                                  \
  JNIEXPORT                                                                   \
  void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_SumForward##DType(    \
      JNIEnv *env, jclass thisClass, JArrayType input, jint inputOffset,       \
      jobjectArray output, jintArray outputOffset, long classPtr)             \
  {                                                                           \
    JNISumUpdateOutput<JArrayType, JType>(env, thisClass, input, inputOffset, \
                                          output, outputOffset, classPtr);    \
  }

#define SumBackward(DType, JType, JArrayType)                               \
  JNIEXPORT                                                                 \
  void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_SumBackward##DType( \
      JNIEnv *env, jclass thisClass, JArrayType inputDiff,                   \
      jint inputDiffOffset, jobjectArray outputDiff,                        \
      jintArray outputDiffOffset, long classPtr)                            \
  {                                                                         \
    JNISumUpdateGradInput<JArrayType, JType>(env, thisClass, inputDiff,     \
                                             inputDiffOffset, outputDiff,   \
                                             outputDiffOffset, classPtr);   \
  }

#define SumNext(DType, JType, JArrayType) \
  JNIEXPORT \
  void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_SetSumNext##DType( \
      JNIEnv *env, jclass thisClass, jlong next, jint index, jlong curr) \
  { \
    JNISumSetNext<JArrayType, JType>(env, thisClass, next, index, curr);\
  }

#ifdef __cplusplus
extern "C" {
#endif

// Double
SumInit(Double, jdouble, jdoubleArray);
SumForward(Double, jdouble, jdoubleArray);
SumBackward(Double, jdouble, jdoubleArray);
SumNext(Double, jdouble, jdoubleArray);

// Float
SumInit(Float, jfloat, jfloatArray);
SumForward(Float, jfloat, jfloatArray);
SumBackward(Float, jfloat, jfloatArray);
SumNext(Float, jfloat, jfloatArray);

JNIEXPORT
void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_SetIPrevFloat(
    JNIEnv *env, jclass thisClass, long prev, int index, long curr)
{
  MKLSum<float> *ptr = reinterpret_cast<MKLSum<float> *>(prev);
  ptr->setIPrev(index, curr);
}

JNIEXPORT
void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_SetIPrevDouble(
    JNIEnv *env, jclass thisClass, long prev, int index, long curr)
{
  MKLSum<double> *ptr = reinterpret_cast<MKLSum<double> *>(prev);
  ptr->setIPrev(index, curr);
}

#ifdef __cplusplus
}

#endif
