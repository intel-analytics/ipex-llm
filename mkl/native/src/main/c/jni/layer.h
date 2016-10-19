#ifndef _MKL_LAYER_H
#define _MKL_LAYER_H
#include <memory>

#include "MKLWrapper.h"
#include "memory.h"
#include "cpu_info.hpp"

template <typename DType>
class MKLLayer
{
 public:
  MKLLayer();
  ~MKLLayer();

  static void setPrev(long prev, long curr);
  static void setNext(long next, long curr);
  // virtual void setIPrev(int index, long curr);
  static void setUseNext(long ptr, int value);

  void init(size_t inputNumber, size_t inputChannel, size_t inputHeight,
            size_t inputWidth, size_t dimension);

  std::shared_ptr<MKLData<DType>> input, output, gradInput, gradOutput;

  int dimension;
  std::string name;

  // parameters of pooling layer
  size_t inputSize[4];
  size_t inputStrides[4];

  // If it's the first pass, we should create some conversions.
  // After that, we need not do that again.
  // Default is true.
  //
  // Note:
  //   1. Defaultly, we assume that the address of input will not change.
  //   2. The address of input is real address of Array in JVM.
  //   3. TODO It will set to false after an iteration (forward and backward).
  bool isFirstPass;

  dnnPrimitive_t forwardPrim, backwardPrim;
};

template <typename DType>
void MKLLayer<DType>::init(size_t inputNumber, size_t inputChannel,
                           size_t inputHeight, size_t inputWidth,
                           size_t dimension)
{
  inputSize[0] = inputWidth;
  inputSize[1] = inputHeight;
  inputSize[2] = inputChannel;
  inputSize[3] = inputNumber;

  this->dimension = dimension;

  inputStrides[0] = 1;
  for (int i = 1; i < 4; i++) {
    inputStrides[i] = inputStrides[i - 1] * inputSize[i - 1];
  }

  input->createUsrLayout(dimension, inputSize, inputStrides);
  gradInput->createUsrLayout(dimension, inputSize, inputStrides);
}

template <typename DType>
MKLLayer<DType>::MKLLayer()
    : input(new MKLData<DType>()),
      output(new MKLData<DType>()),
      gradInput(new MKLData<DType>()),
      gradOutput(new MKLData<DType>()),
      isFirstPass(true),
      forwardPrim(NULL),
      backwardPrim(NULL)
{
}

template <typename DType>
MKLLayer<DType>::~MKLLayer()
{
  if (forwardPrim) {
    dnnDelete<DType>(forwardPrim);
    forwardPrim = NULL;
  }

  if (backwardPrim) {
    dnnDelete<DType>(backwardPrim);
    backwardPrim = NULL;
  }
}

template <typename DType>
void MKLLayer<DType>::setPrev(long prev, long curr)
{
  MKLLayer<DType> *prevLayer = reinterpret_cast<MKLLayer<DType> *>(prev);
  MKLLayer<DType> *currLayer = reinterpret_cast<MKLLayer<DType> *>(curr);

#if 0
//  dnnLayout_t prevLayout = prevLayer->gradOutput->getMklLayout();
//  dnnLayout_t currLayout = currLayer->gradInput->getMklLayout();
//
//  if (dnnLayoutCompare<DType>(prevLayout, currLayout)) {
//    prevLayer->gradOutput->setUseNext(true);
//    prevLayer->gradOutput->setMklData(currLayer->gradInput->getData(),
//                                      currLayer->gradInput->getUsrData() !=
//                                      currLayer->gradInput->getMklData());
//    currLayer->gradInput->setUsePrev(true);
//  } else {
//    LOG(DBG) << "The layout is not the same";
//  }
#endif

  if (prevLayer && prevLayer->output->getMklData()) {
    dnnLayout_t prevLayout = prevLayer->output->getMklLayout();
    dnnLayout_t currLayout = currLayer->input->getMklLayout();

    currLayer->input->layoutPrev = prevLayout;
    void *dataMkl = prevLayer->output->getMklData();
    currLayer->input->dataPrev = dataMkl;

    if (currLayer->input->getMklData()) {
      dnnReleaseBuffer<DType>(currLayer->input->getMklLayout());
      currLayer->input->setMklData(NULL);
    }

    currLayer->input->setUsePrev(true);
    prevLayer->output->setUseNext(true);
  }

#if 0
//  prevLayout = prevLayer->gradOutput->getMklLayout();
//  currLayout = currLayer->gradInput->getMklLayout();
//
//  if (currLayout)
//    prevLayer->gradOutput->setMklLayout(currLayout);
//  if (currLayer->gradInput->getMklData()) {
//    void *dataMkl = currLayer->gradInput->getMklData();
//    prevLayer->gradOutput->setMklData(data, true);
//
//    prevLayer->gradOutput->setUseNext(true);
//    currLayer->gradInput->setUsePrev(true);
//  }
#endif

#if 0
//  if (dnnLayoutCompare<DType>(prevLayout, currLayout)) {
//    prevLayer->output->setUseNext(true);
//    currLayer->input->setMklData(prevLayer->output->getData(),
//                                 prevLayer->output->getUsrData() !=
//                                 prevLayer->output->getMklData());
//    currLayer->input->setUsePrev(true);
//  } else {
//    LOG(DBG) << "The layout is not the same";
//  }
#endif
}

template <typename DType>
void MKLLayer<DType>::setNext(long next, long curr)
{
  MKLLayer<DType> *nextLayer = reinterpret_cast<MKLLayer<DType> *>(next);
  MKLLayer<DType> *currLayer = reinterpret_cast<MKLLayer<DType> *>(curr);

  //LOG(DBG) << "nextLayer = " << nextLayer;
  //LOG(DBG) << "currLayer = " << currLayer;

  if (nextLayer && nextLayer->gradInput->getMklData()) {
    currLayer->gradOutput->layoutNext = nextLayer->gradInput->getMklLayout();
    currLayer->gradOutput->dataNext = nextLayer->gradInput->getMklData();

    if (currLayer->gradOutput->getMklData()) {
      dnnReleaseBuffer<DType>(currLayer->gradOutput->getMklData());
      currLayer->gradOutput->setMklData(NULL);
    }

    currLayer->gradOutput->setUseNext(true);
    nextLayer->gradInput->setUsePrev(true);
  }
}

template <typename DType>
void MKLLayer<DType>::setUseNext(long modulePtr, int value)
{
  MKLLayer<DType> *layer = reinterpret_cast<MKLLayer<DType>*>(modulePtr);
  bool v = false;
  if (value > 0) v = true;

  if (layer) { layer->output->setUseNext(v); }
}

#endif
