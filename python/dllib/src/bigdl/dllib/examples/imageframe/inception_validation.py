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

from bigdl.util.common import *

from bigdl.transform.vision.image import *

from bigdl.optim.optimizer import *

from pyspark import SparkContext

from bigdl.nn.layer import *

from optparse import OptionParser

import sys

parser = OptionParser()
parser.add_option("-f", "--folder", type=str, dest="folder", default="",
                  help="url of hdfs folder store the hadoop sequence files")
parser.add_option("--model", type=str, dest="model", default="", help="model path")
parser.add_option("-b", "--batchSize", type=int, dest="batchSize", default=0, help="total batch size")
def get_data(url, sc=None, data_type="val"):
    path = os.path.join(url, data_type)
    return SeqFileFolder.files_to_image_frame(url=path, sc=sc, class_num=1000)

def run(image_path, model_path, batch_size):
    sparkConf = create_spark_conf().setAppName("test_validation")
    sc = get_spark_context(sparkConf)
    init_engine()
    transformer = Pipeline([PixelBytesToMat(), Resize(256, 256), CenterCrop(224, 224),
                            ChannelNormalize(123.0, 117.0, 104.0),
                            MatToTensor(), ImageFrameToSample(input_keys=["imageTensor"],
                                target_keys=["label"])])
    raw_image_frame = get_data(image_path, sc)
    transformed = transformer(raw_image_frame)
    model = Model.loadModel(model_path)
    result = model.evaluate(transformed, int(batch_size), [Top1Accuracy()])
    print "top1 accuray", result[0]

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "parameters needed : <imagePath> <modelPath> <batchSize>"
    image_path = sys.argv[1]
    model_path = sys.argv[2]
    batch_size = sys.argv[3]
    run(image_path, model_path, batch_size)
