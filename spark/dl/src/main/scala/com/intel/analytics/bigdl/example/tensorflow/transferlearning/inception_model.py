import tensorflow as tf
import os
from nets.inception import inception_v1
from nets.inception import inception_v1_arg_scope

slim = tf.contrib.slim


class InceptionModel(object):

    def __init__(self, trainable_scopes=None, checkpoint_exclude_scopes=None, checkpoints_file=None):
        if trainable_scopes is None:
            self.trainable_scopes = []
        else:
            self.trainable_scopes=trainable_scopes

        if checkpoint_exclude_scopes is None:
            self.checkpoint_exclude_scopes = []
        else:
            self.checkpoint_exclude_scopes = checkpoint_exclude_scopes
        self.checkpoints_file = checkpoints_file

        self.init_fn = None

    def _get_init_fn(self):

        exclusions = [scope.strip() for scope in self.checkpoint_exclude_scopes]

        variables_to_restore = []
        for var in slim.get_model_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)

        return slim.assign_from_checkpoint_fn(self.checkpoints_file, variables_to_restore)

    def _load_batch(self, dataset, batch_size=32, height=224, width=224):

        data_provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset, common_queue_capacity=32,
            common_queue_min=8)
        image_raw, label = data_provider.get(['image', 'label'])

        # Preprocess image for usage by Inception.
        image = self._preprocess(image_raw, height, width)

        # Batch it up.
        images, labels = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=1,
            capacity=2 * batch_size)

        return images, labels

    def _preprocess(self, image, height, width, scope=None):
        with tf.name_scope(scope, 'eval_image', [image, height, width]):
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            # Crop the central region of the image with an area containing 87.5% of
            # the original image.
            # if central_fraction:
            #   image = tf.image.central_crop(image, central_fraction=central_fraction)

            if height and width:
                # Resize the image to the specified height and width.
                image = tf.expand_dims(image, 0)
                image = tf.image.resize_bilinear(image, [height, width],
                                                 align_corners=False)
                image = tf.squeeze(image, [0])
            image = tf.subtract(image, 0.5)
            image = tf.multiply(image, 2.0)
            return image

    def build(self, dataset, image_height, image_width, num_classes, is_training):

        images, labels = self._load_batch(dataset, height=image_height, width=image_width)

        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception_v1_arg_scope()):
            logits, end_points = inception_v1(images, num_classes=num_classes, is_training=is_training)

        # Specify the loss function:
        one_hot_labels = slim.one_hot_encoding(labels, num_classes)
        loss = slim.losses.softmax_cross_entropy(logits, one_hot_labels)

        # Create some summaries to visualize the training process:
        tf.summary.scalar('losses/Total Loss', loss)

        if is_training:
            # Specify the optimizer and create the train op:
            optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01)
            variables_to_train = []
            for scope in self.trainable_scopes:
                variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                variables_to_train.extend(variables)

            variables = slim.get_model_variables("InceptionV1/Logits")
            exec_op = slim.learning.create_train_op(loss, optimizer, variables_to_train=variables)

        else:
            exec_op = end_points['Predictions']

        if self.checkpoints_file is not None:
            self.init_fn = self._get_init_fn()

        return exec_op, tf.get_default_graph()


