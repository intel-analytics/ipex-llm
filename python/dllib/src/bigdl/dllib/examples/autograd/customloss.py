#
# Copyright 2018 Analytics Zoo Authors.
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

from zoo.common.nncontext import *
from zoo.pipeline.api.autograd import *
from zoo.pipeline.api.keras.layers import *
from zoo.pipeline.api.keras.models import *


def mean_absolute_error(y_true, y_pred):
    result = mean(abs(y_true - y_pred), axis=1)
    return result


if __name__ == "__main__":
    sc = init_nncontext("customloss example")
    data_len = 1000
    X_ = np.random.uniform(0, 1, (1000, 2))
    Y_ = ((2 * X_).sum(1) + 0.4).reshape([data_len, 1])
    model = Sequential()
    model.add(Dense(1, input_shape=(2,)))
    model.compile(optimizer=SGD(learningrate=1e-2),
                  loss=mean_absolute_error,
                  metrics=None)
    model.fit(x=X_,
              y=Y_,
              batch_size=32,
              nb_epoch=500,
              validation_data=None,
              distributed=False)
    w = model.get_weights()
    print(w)
    pred = model.predict_local(X_)
    print("finished...")
    sc.stop()
