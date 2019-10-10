from bigdl.nn.layer import *
from bigdl.util.common import *
import numpy as np

sc = SparkContext(appName="test", conf=create_spark_conf())
redire_spark_logs()
show_bigdl_info_logs()
init_engine()

input1 = Input()
lstm = Recurrent().add(LSTM(100, 100))(input1)
model = Model([input1],[lstm])
# model.to_staticgraph()
model.set_input_formats([27])
model.set_output_formats([27])
model.to_irgraph()

input = np.random.rand(5, 50, 100)
gradOutput = np.random.rand(5, 50, 100)

for i in range(100):
    output = model.forward(input)
    gradInput = model.backward(input, gradOutput)