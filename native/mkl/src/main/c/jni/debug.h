#ifndef _DEBUG_H_
#define _DEBUG_H_

#include <stdio.h>

//#define CHECK_EQ(x, y) CHECK((x) == (y))
#define CHECK_EQ(x, y)                              \
  if (!((x) == (y)))                                \
    printf("[MKL] %s]:%d check error\n", __FILE__, __LINE__);

#define CHECK_NE(x, y)                              \
  if (!((x) != (y)))                                \
    printf("[MKL] %s]:%d check error\n", __FILE__, __LINE__);

#endif
