import paddle.fluid as fluid
from functools import partial
import numpy as np
from paddle.fluid.layer_helper import LayerHelper

def get_inplanes():
    return [64, 128, 256, 512]


def conv1x1x1(in_planes, out_planes, stride=1):
    return fluid.dygraph.Conv3D(in_planes,
                                 out_planes,
                                 filter_size=1,
                                 stride=stride,
                                 param_attr = fluid.initializer.MSRAInitializer(uniform=False),
                                 bias_attr = False)


def conv3x3x3(in_planes, out_planes, stride=1):
    return fluid.dygraph.Conv3D(in_planes,
                                 out_planes,
                                 filter_size=3,
                                 stride=stride,
                                 padding=1,
                                 param_attr = fluid.initializer.MSRAInitializer(uniform=False),
                                 bias_attr = False)


def pool3d(x):
    return fluid.layers.pool3d(x,pool_size=3, pool_stride=2, pool_padding=1)


class BasicBlock(fluid.dygraph.Layer):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock,self).__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = fluid.dygraph.BatchNorm(planes)
        self.relu = fluid.layers.relu()
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = fluid.dygraph.BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(fluid.dygraph.Layer):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = fluid.dygraph.BatchNorm(planes,
                                           param_attr=fluid.initializer.ConstantInitializer(value=1),
                                           bias_attr=fluid.initializer.ConstantInitializer(value=0))
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = fluid.dygraph.BatchNorm(planes,
                                           param_attr=fluid.initializer.ConstantInitializer(value=1),
                                           bias_attr=fluid.initializer.ConstantInitializer(value=0))
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = fluid.dygraph.BatchNorm(planes * self.expansion,
                                           param_attr=fluid.initializer.ConstantInitializer(value=1),
                                           bias_attr=fluid.initializer.ConstantInitializer(value=0))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = fluid.layers.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = fluid.layers.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = fluid.layers.relu(out)

        return out


class Resnet(fluid.dygraph.Layer):
    #定义网络结构，代码补齐
    def __init__(self,
                 block,
                layers,
                block_inplanes,
                n_input_channels=3,
                conv1_t_size=7,
                conv1_t_stride=1,
                no_max_pool=False,
                shortcut_type='B',
                widen_factor=1.0,
                n_classes=1039):

        super(Resnet,self).__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = fluid.dygraph.Conv3D(num_channels = n_input_channels,
                               num_filters = self.in_planes,
                               filter_size  = (conv1_t_size, 7, 7),
                               stride = (conv1_t_stride, 2, 2),
                               padding = (conv1_t_size // 2, 3, 3),
                               param_attr = fluid.initializer.MSRAInitializer(uniform=False),
                               bias_attr = False)
        self.bn1 = fluid.dygraph.BatchNorm(self.in_planes,
                                           param_attr=fluid.initializer.ConstantInitializer(value=1),
                                           bias_attr=fluid.initializer.ConstantInitializer(value=0))
        #self.relu = fluid.layers.relu()
        #self.maxpool = fluid.layers.pool3d(pool_size=3, pool_stride=2, pool_padding=1)
        # self.block = Bottleneck(block_inplanes[0],layers[0])
        self.block = block
        self.layer1 = self._make_layer(self.block, block_inplanes[0], layers[0])
        self.layer2 = self._make_layer(self.block, block_inplanes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(self.block, block_inplanes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(self.block, block_inplanes[3], layers[3], stride=2)

        self.fc = fluid.dygraph.Linear(block_inplanes[3] * self.block.expansion, n_classes)

    def _downsample_basic_block(self, x, planes, stride):
        out = fluid.layers.pool3d(x, pool_size=1, pool_stride=stride)
        zero_pads = fluid.layers.zeros(out.size(0), planes - out.size(1), out.size(2),out.size(3), out.size(4))
        out = fluid.layers.concat([out.data, zero_pads], axis=1)
        return out

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = fluid.dygraph.Sequential(conv1x1x1(self.in_planes, planes * block.expansion, stride),
                        fluid.dygraph.BatchNorm(planes * block.expansion,
                                                  param_attr=fluid.initializer.ConstantInitializer(value=1),
                                                  bias_attr=fluid.initializer.ConstantInitializer(value=0)))
        layers = []
        layers.append(
            Bottleneck(self.in_planes,
                  planes,
                  stride,
                  downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.in_planes, planes))

        return fluid.dygraph.Sequential(*layers)

    def forward(self, x, label=None):
        # x = fluid.layers.reshape(x, [-1, x.shape[2], x.shape[3], x.shape[4]])
        x = fluid.layers.reshape(x, [-1, 3, int(x.shape[2]/3), x.shape[3], x.shape[4]])
        x = self.conv1(x)
        x = self.bn1(x)
        x = fluid.layers.relu(x)
        # x = self.relu(x)
        if not self.no_max_pool:
            x = pool3d(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = fluid.layers.adaptive_pool3d(x, pool_size=(1, 1, 1))
        y = fluid.layers.reshape(x, [x.shape[0], -1])
        y = self.fc(y)
        out = fluid.layers.softmax(y, axis=-1)

        if label is not None:
            acc = fluid.layers.accuracy(input=out, label=label)
            return out, acc
        else:
            return out

def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = Resnet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = Resnet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = Resnet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = Resnet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = Resnet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = Resnet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = Resnet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)
    return model

if __name__ == '__main__':
    with fluid.dygraph.guard():
        network = Resnet(BasicBlock, [2,2,2,2], get_inplanes())
        img = np.zeros([1, 10, 3, 224, 224]).astype('float32')
        img = fluid.dygraph.to_variable(img)
        outs = network(img).numpy()
        print(outs)