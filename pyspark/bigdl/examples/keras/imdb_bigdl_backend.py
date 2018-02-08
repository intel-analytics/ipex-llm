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


# Alternative of the example imdb_cnn_lstm.py using BigDL backend
# You can essentially employ the same code written in Keras 1.2.2 and add the following line
# after building, compiling the model and before the training.
#     model = with_bigdl_backend(model)

from bigdl.keras.backend import *
from bigdl.examples.keras.imdb_cnn_lstm import *

X_train, y_train, X_test, y_test = load_imdb()

# Define a model in Keras 1.2.2 and compile
keras_model = build_keras_model()
keras_model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

# Add this line of code to use BigDL backend alternatively
model = with_bigdl_backend(keras_model)

model.fit(X_train, y_train, batch_size=32, nb_epoch=1,
          validation_data=(X_test, y_test), is_distributed=True)
# You may need to use data different from validation to evaluate or predict the model.
accuracy = model.evaluate(X_test, y_test, batch_size=32, is_distributed=True)
print("Top1Accuracy: ", accuracy[0])
results = model.predict(X_test, is_distributed=True)
print("First predict result: ", results.take(1))
