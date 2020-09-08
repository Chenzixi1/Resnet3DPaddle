import torch
import paddle.fluid as fluid
from collections import OrderedDict
from model import ResNet3D

torch_weight = torch.load('r3d50_KM_200ep.pth', map_location=torch.device('cpu'))
# 全连接1039
weight = []
for torch_key in torch_weight['state_dict'].keys():
    weight.append([torch_key,torch_weight['state_dict'][torch_key].detach().numpy()])

with fluid.dygraph.guard():
    # 加载网络结构
    paddle_model = ResNet3D.generate_model(50)

    # 读取paddle网络结构的参数列表
    paddle_weight = paddle_model.state_dict()
    # 将paddle权重的参数列表打印出来
    # for paddle_key in paddle_weight:
    #     print(paddle_key)

    # 进行模型参数转换
    new_weight_dict = OrderedDict()

    i = 0
    for paddle_key in paddle_weight.keys():
        print(paddle_key)
        if 'num_batches_tracked' in weight[i][0]:
            i += 1
        if weight[i][0].find('fc'):
            new_weight_dict[paddle_key] = weight[i][1]
        else:
            new_weight_dict[paddle_key] = weight[i][1].T
        i += 1

    paddle_model.set_dict(new_weight_dict)
    fluid.dygraph.save_dygraph(paddle_model.state_dict(),'model/res50')