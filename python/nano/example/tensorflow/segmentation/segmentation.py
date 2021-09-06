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

# This file is adapted from
# https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tutorials/images/segmentation.ipynb
# Copyright 2019 The TensorFlow Authors.

##### Copyright 2019 The TensorFlow Authors.

# Licensed under the Apache License, Version 2.0 (the "License");


#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tensorflow as tf

from tensorflow_examples.models.pix2pix import pix2pix
from bigdl.nano.tf.keras import Model as Model

import tensorflow_datasets as tfds
tfds.disable_progress_bar()


def main():
    dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

    def normalize(input_image, input_mask):
        input_image = tf.cast(input_image, tf.float32) / 255.0
        input_mask -= 1
        return input_image, input_mask

    @tf.function
    def load_image_train(datapoint):
        input_image = tf.image.resize(datapoint['image'], (128, 128))
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)

        input_image, input_mask = normalize(input_image, input_mask)

        return input_image, input_mask

    def load_image_test(datapoint):
        input_image = tf.image.resize(datapoint['image'], (128, 128))
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

        input_image, input_mask = normalize(input_image, input_mask)

        return input_image, input_mask

    TRAIN_LENGTH = info.splits['train'].num_examples
    BATCH_SIZE = 64
    BUFFER_SIZE = 1000
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

    train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test = dataset['test'].map(load_image_test)

    train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_dataset = test.batch(BATCH_SIZE)

    OUTPUT_CHANNELS = 3

    base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

    # 使用这些层的激活设置
    layer_names = [
        'block_1_expand_relu',  # 64x64
        'block_3_expand_relu',  # 32x32
        'block_6_expand_relu',  # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',  # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # 创建特征提取模型
    down_stack = Model(inputs=base_model.input, outputs=layers)

    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),  # 32x32 -> 64x64
    ]

    def unet_model(output_channels):
        inputs = tf.keras.layers.Input(shape=[128, 128, 3])
        x = inputs

        # 在模型中降频取样
        skips = down_stack(x)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # 升频取样然后建立跳跃连接
        for up, skip in zip(up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        # 这是模型的最后一层
        last = tf.keras.layers.Conv2DTranspose(
            output_channels, 3, strides=2,
            padding='same')  # 64x64 -> 128x128

        x = last(x)

        return Model(inputs=inputs, outputs=x)

    model = unet_model(OUTPUT_CHANNELS)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    tf.keras.utils.plot_model(model, show_shapes=True)

    def create_mask(pred_mask):
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]
        return pred_mask[0]

    EPOCHS = 20
    VAL_SUBSPLITS = 5
    VALIDATION_STEPS = info.splits['test'].num_examples // BATCH_SIZE // VAL_SUBSPLITS

    model_history = model.fit(train_dataset, epochs=EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_steps=VALIDATION_STEPS,
                              validation_data=test_dataset)

    model.evaluate(test_dataset)

if __name__ == "__main__":
    main()
