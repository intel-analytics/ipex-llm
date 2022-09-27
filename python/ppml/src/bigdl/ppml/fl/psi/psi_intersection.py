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
import threading

from bigdl.ppml.utils.log4Error import invalidOperationError


class PsiIntersection(object):
    def __init__(self, max_collection=1) -> None:
        self.intersection = []
        self._thread_intersection = []

        self.max_collection = int(max_collection)
        self.condition = threading.Condition()
        self._lock = threading.Lock()

        self.collection = []
        
    def find_intersection(self, a, b):
        return list(set(a) & set(b))

    def add_collection(self, collection):
        with self._lock:
            invalidOperationError(len(self.collection) < self.max_collection,
                f"PSI collection is full, got: {len(self.collection)}/{self.max_collection}")
            self.collection.append(collection)
            logging.debug(f"PSI got collection {len(self.collection)}/{self.max_collection}")
            if len(self.collection) == self.max_collection:
                current_intersection = self.collection[0]
                for i in range(1, len(self.collection)):
                    current_intersection = \
                        self.find_intersection(current_intersection, self.collection[i])
                self.intersection = current_intersection
                self.collection.clear()

    def get_intersection(self):
        with self._lock:
            return self.intersection
