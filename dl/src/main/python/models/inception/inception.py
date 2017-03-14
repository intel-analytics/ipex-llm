from nn.layer import *

def scala_T(input_T):
    input_T.insert(0,0) # insert into index 0 spot, such that the real data starts from index 1
    return input_T
    
    
def Inception_Layer_v1(input_size, config, name_prefix = ""):
    concat = Concat(2)
    conv1 = Sequential()
    conv1.add(SpatialConvolution(input_size,
                                 config[1][1], 1, 1, 1, 1, init_method="Xavier")) #setName(name_prefix + "1x1")
    conv1.add(ReLU(True)) #.setName(name_prefix + "relu_1x1")
    concat.add(conv1)
    conv3 = Sequential()
    conv3.add(SpatialConvolution(input_size,
                                 config[2][1], 1, 1, 1, 1, init_method="Xavier"))#.setInitMethod(Xavier).setName(name_prefix + "3x3_reduce"))
    conv3.add(ReLU(True).setName(name_prefix + "relu_3x3_reduce"))
    conv3.add(SpatialConvolution(config[2][1],
                                 config[2][2], 3, 3, 1, 1, 1, 1, init_method="Xavier"))#.setInitMethod(Xavier).setName(name_prefix + "3x3"))
    conv3.add(ReLU(True))#.setName(name_prefix + "relu_3x3"))
    concat.add(conv3)
    conv5 = Sequential()
    conv5.add(SpatialConvolution(input_size,
                                 config[3][1], 1, 1, 1, 1, init_method="Xavier"))#.setInitMethod(Xavier).setName(name_prefix + "5x5_reduce" ))
    conv5.add(ReLU(True))#.setName(name_prefix + "relu_5x5_reduce"))
    conv5.add(SpatialConvolution(config[3][1],
                                 config[3][2], 5, 5, 1, 1, 2, 2, init_method="Xavier"))#.setInitMethod(Xavier).setName(name_prefix + "5x5"))
    conv5.add(ReLU(True))#.setName(name_prefix + "relu_5x5"))
    concat.add(conv5)
    pool = Sequential()
    pool.add(SpatialMaxPooling(3, 3, 1, 1, 1, 1, to_ceil=True) #.setName(name_prefix + "pool"))
    pool.add(SpatialConvolution(input_size,
                                config[4][1], 1, 1, 1, 1, init_method="Xavier"))#.setName(name_prefix + "pool_proj")
    pool.add(ReLU(True)) #ReLU.setName(name_prefix + "relu_pool_proj")
    concat.add(pool))#.setName(name_prefix + "output")
    return concat
    
class Inception_v1_NoAuxClassifier:

        
    def __init__(self):
        self.model = Sequential()
    

    def build_model(self, class_num):
        self.model.add(SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1, False,"Xavier"))
        self.model.add()
        self.model.add(ReLU(True))
        self.model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True))
        self.model.add(SpatialCrossMapLRN(5, 0.0001, 0.75))
        self.model.add(SpatialConvolution(64, 64, 1, 1, 1, 1,"Xavier"))
        self.model.add(ReLU(True))
        self.model.add(SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1,"Xavier"))
        self.model.add(ReLU(True))
        self.model.add(SpatialCrossMapLRN(5, 0.0001, 0.75))
        self.model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True))
        self.model.add(Inception_Layer_v1(192, scala_T([scala_T([64]), scala_T([96, 128]), scala_T([16, 32]), scala_T([32])]), "inception_3a/"))
        self.model.add(Inception_Layer_v1(256, scala_T([scala_T([128]), scala_T([128, 192]), scala_T([32, 96]), scala_T([64])]), "inception_3b/"))
        self.model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True))
        self.model.add(Inception_Layer_v1(480, scala_T([scala_T([192]), scala_T([96, 208]), scala_T([16, 48]), scala_T([64])]), "inception_4a/"))
        self.model.add(Inception_Layer_v1(512, scala_T([scala_T([160]), scala_T([112, 224]), scala_T([24, 64]), scala_T([64])]), "inception_4b/"))
        self.model.add(Inception_Layer_v1(512, scala_T([scala_T([128]), scala_T([128, 256]), scala_T([24, 64]), scala_T([64])]), "inception_4c/"))
        self.model.add(Inception_Layer_v1(512, scala_T([scala_T([112]), scala_T([144, 288]), scala_T([32, 64]), scala_T([64])]), "inception_4d/"))
        self.model.add(Inception_Layer_v1(528, scala_T([scala_T([256]), scala_T(1[60, 320]), scala_T([32, 128]), scala_T([128])]), "inception_4e/"))
        self.model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True))
        self.model.add(Inception_Layer_v1(832, scala_T([scala_T([256]), scala_T([160, 320]), scala_T([32, 128]), scala_T([128])]), "inception_5a/"))
        self.model.add(Inception_Layer_v1(832, scala_T([scala_T([384]), scala_T([192, 384]), scala_T([48, 128]), scala_T([128])]), "inception_5b/"))
        self.model.add(SpatialAveragePooling(7, 7, 1, 1))
        self.model.add(Dropout(0.4))
        self.model.add(View(1024, num_input_dims=3))
        self.model.add(Linear(1024, classNum,"Xavier"))
        self.model.add(LogSoftMax())
        self.model.reset()
        return self.model