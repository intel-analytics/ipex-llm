/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
