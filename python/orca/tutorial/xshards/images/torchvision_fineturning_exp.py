# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
import os
from PIL import Image
from bigdl.orca.learn.pytorch.callbacks import MainCallback
from bigdl.orca.learn.tf2.estimator import Estimator
import numpy as np
import torch
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.data import read_images
from bigdl.orca.data.shard import SparkXShards
import torchvision
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class CustomMainCB(MainCallback):
    def on_iter_forward(self, runner):
        image, target = runner.batch
        runner.output = runner.model(image, target)
        runner.loss = sum(loss for loss in runner.output.values())

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)



def main():
    sc = init_orca_context(cluster_mode="local", cores=4, memory="6g")
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    input_path = "/Users/guoqiong/intelWork/data/PennFudanPed/"
    data_shards = read_images(file_path=input_path + "PNGImages",
                              target_path=input_path + "PedMasks",
                              image_type=".png",
                              target_type=".png")
    collected = data_shards.collect()

    def get_target(image):
        img, mask = image
        img = img.convert("RGB")
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        return img, target

    def get_transform(image):
        # transforms.append(T.PILToTensor())
        # transforms.append(T.ConvertImageDtype(torch.float))
        # if train:
        #     transforms.append(T.RandomHorizontalFlip(0.5))
        features = image[0]
        features = T.PILToTensor()(features)
        features = T.ConvertImageDtype(torch.float)(features)

        # features = features.numpy()
        # features = np.array(features)
        return features, image[1]

    def stack_feature_labels(data):
        def per_partition(iterator):
            targets = {}
            features = []
            keys = ["boxes", "labels", "masks"]
            for key in keys:
                targets[key] = []
            for it in iterator:
                feature, target = it[0], it[1]
                features.append(feature)
                for key in keys:
                    targets[key].append(target[key])
            out = {'x': np.array(features),
                   'y': targets}  # dictionary of list
            print("in stack_feature_label", np.array(features).shape)
            return [out]
        rdd = data.rdd.mapPartitions(lambda x: per_partition(x))
        return SparkXShards(rdd)

    data_shards = data_shards.transform_shard(get_target)
    data_shards = data_shards.transform_shard(get_transform)
    data_shards = stack_feature_labels(data_shards)

    def collate_fn(batch):
        out = tuple(zip(*batch))
        out0 = np.array(out[0])
        return out0, out[1]

    config = {"collate_fn": collate_fn}

    def model_creator(config):
        model = get_model_instance_segmentation(2)
        model.to(device)
        return model

    def optimizer_creator(model, config):
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)

        return optimizer

    orca_estimator = Estimator.from_torch(model=model_creator,
                                          optimizer=optimizer_creator,
                                          backend="spark",
                                          config=config)

    orca_estimator.fit(data=data_shards,
                       batch_size=4,
                       epochs=1,
                       callbacks=[CustomMainCB()])

    print("That's it!")


if __name__ == '__main__':
    main()