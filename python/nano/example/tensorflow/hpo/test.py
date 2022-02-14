
import tensorflow as tf

from tensorflow.keras.layers import Flatten

import bigdl.nano.automl.hpo as hpo
import bigdl.nano.automl.hpo.space as space

# decorate the layer class to accept automl.hpo.space as input argument
@hpo.obj()
class Dense(tf.keras.layers.Dense):
    pass
#@hpo.obj()
#class Adam(tf.keras.optimizers.Adam):
#    pass
#myoptim = Adam(learning_rate=space.Real(1e-2, 1e-1, log=True), beta_2=space.Real(1e-5, 1e-3, log=True))
#print(myoptim.kwspaces)
#config={'learning_rate': 0.008, 'beta_2':0.0002}
#opt = myoptim.sample(**config)

dense = Dense(units=space.Int(2,3), activation='softmax')
print(dense.kwspaces)

config={'units':10}
d2=dense.sample(**config)




