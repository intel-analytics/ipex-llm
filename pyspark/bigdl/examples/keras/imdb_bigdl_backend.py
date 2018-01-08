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


# Alternative of the example imdb_cnn_lstm.py with BigDL backend
# You can essentially employ the same code written in Keras 1.2.2 and add the following line
# after building the model and before training the model.
#     model = with_bigdl_backend(model)

from bigdl.keras.backend import *
from bigdl.examples.keras.imdb_cnn_lstm import *

X_train, y_train, X_test, y_test = load_imdb()
model = build_keras_model()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# add this line of code to use BigDL backend alternatively
model = with_bigdl_backend(model)

model.fit(X_train, y_train, batch_size=32, nb_epoch=2, is_distributed=True,
          validation_data=(X_test, y_test))
