#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging

logger = logging.getLogger(__name__)


def outputUserMessage(errMsg, fixMsg=None):
    logger.error(f"\n\n****************************Usage Error************************\n" + errMsg)
    if fixMsg:
        logger.error(f"\n\n****************************How to fix*************************\n"
                     + fixMsg)
    logger.error(f"\n\n****************************Call Stack*************************")


def invalidInputError(condition, errMsg, fixMsg=None):
    if not condition:
        outputUserMessage(errMsg, fixMsg)
        raise RuntimeError(errMsg)


def invalidOperationError(condition, errMsg, fixMsg=None, cause=None):
    if not condition:
        outputUserMessage(errMsg, fixMsg)
        if cause:
            raise cause
        else:
            raise RuntimeError(errMsg)
