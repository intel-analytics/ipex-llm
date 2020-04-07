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

from optparse import OptionParser

from zoo.pipeline.inference import InferenceModel
from zoo.common.nncontext import init_nncontext
from zoo.feature.image import *
from zoo.pipeline.nnframes import *

BATCH_SIZE = 4


def predict(model_path, img_path):
    model = InferenceModel()
    model.load_openvino(model_path,
                        weight_path=model_path[:model_path.rindex(".")] + ".bin",
                        batch_size=BATCH_SIZE)
    sc = init_nncontext("OpenVINO Python resnet_v1_50 Inference Example")
    # pre-processing
    infer_transformer = ChainedPreprocessing([ImageBytesToMat(),
                                             ImageResize(256, 256),
                                             ImageCenterCrop(224, 224),
                                             ImageMatToTensor(format="NHWC", to_RGB=True)])
    image_set = ImageSet.read(img_path, sc).\
        transform(infer_transformer).get_image().collect()
    image_set = np.expand_dims(image_set, axis=1)

    for i in range(len(image_set) // BATCH_SIZE + 1):
        index = i * BATCH_SIZE
        # check whether out of index
        if index >= len(image_set):
            break
        batch = image_set[index]
        # put 4 images in one batch
        for j in range(index + 1, min(index + BATCH_SIZE, len(image_set))):
            batch = np.vstack((batch, image_set[j]))
        batch = np.expand_dims(batch, axis=0)
        # predict batch
        predictions = model.predict(batch)
        result = predictions[0]

        # post-processing for Top-1
        print("batch_" + str(i))
        for r in result:
            output = {}
            max_index = np.argmax(r)
            output["Top-1"] = str(max_index)
            print("* Predict result " + str(output))
    print("finished...")
    sc.stop()

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--image", type=str, dest="img_path",
                      help="The path where the images are stored, "
                           "can be either a folder or an image path")
    parser.add_option("--model", type=str, dest="model_path",
                      help="Zoo Model Path")

    (options, args) = parser.parse_args(sys.argv)
    predict(options.model_path, options.img_path)
