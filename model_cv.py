import copy
import torch
import numpy as np
from collections import OrderedDict
from collections import OrderedDict
from server import *
class Federation:

    def __init__(self, global_parameters,rate):
        self.global_parameters=global_parameters

        self.rate = rate
        self.model_rate=rate

    def get_model_params_with_metadata(self, model):
        params_with_metadata = OrderedDict()  # 使用 OrderedDict 保证顺序

        for name, param in model.named_parameters():
            params_with_metadata[name] = param.clone().detach()  # 直接存储 tensor

        return params_with_metadata

    # def split_model(self, user_idx):  # 为每个客户端分配不同的全局子模型
    #     idx_i = [None for _ in range(len(user_idx))]
    #     idx = [OrderedDict() for _ in range(len(user_idx))]
    #     p = 0
    #
    #     for k, v in self.global_parameters.items():
    #         p += 1
    #         parameter_type = k.split('.')[-1]
    #         for m in range(len(user_idx)):
    #             if 'weight' in parameter_type or 'bias' in parameter_type:
    #                 if parameter_type == 'weight':
    #                     if v.dim() > 1:  # 对于多维权重（卷积层、全连接层等）
    #                         input_size = v.size(1)
    #                         output_size = v.size(0)
    #
    #                         # 处理卷积层（conv1 和 conv2）
    #                         if 'conv1' in k or 'conv2' in k:
    #                             if idx_i[m] is None:
    #                                 idx_i[m] = torch.arange(input_size, device=v.device)
    #                             input_idx_i_m = idx_i[m]
    #                             scaler_rate = self.model_rate[user_idx[m]]
    #                             local_output_size = int(np.ceil(output_size * scaler_rate))
    #                             output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]
    #                             idx_i[m] = output_idx_i_m
    #                             idx[m][k] = (output_idx_i_m, input_idx_i_m)
    #
    #                         # 处理 downsample 层（shortcut）
    #                         # elif 'downsample' in k:
    #                         #     # 直接查找对应的 conv1 或者其它卷积层（不只是替换名称）
    #                         #     # if 'conv' in k:  # 例如 "downsample.0" 对应 "conv1"
    #                         #     #     input_idx_i_m = idx[m].get(k.replace('downsample', 'conv1'), (None, None))[1]
    #                         #     # else:
    #                         #     #     # 对于其它的 downsample 层，根据实际情况处理
    #                         #     #     input_idx_i_m = idx[m].get(k.replace('downsample', 'conv1'), (None, None))[1]
    #                         #     input_idx_i_m = idx[m].get(k.replace('downsample', 'conv1'), (None, None))[1]
    #                         #
    #                         #     output_idx_i_m = idx_i[m]
    #                         #     idx[m][k] = (output_idx_i_m, input_idx_i_m)
    #
    #                         # 处理全连接层（fc）
    #                         elif 'fc' in k:
    #                             input_idx_i_m = idx_i[m]
    #                             output_idx_i_m = torch.arange(output_size, device=v.device)
    #                             idx[m][k] = (output_idx_i_m, input_idx_i_m)
    #                     else:  # 对于一维权重（通常是 fc 层的权重）
    #                         input_idx_i_m = idx_i[m]
    #                         idx[m][k] = input_idx_i_m
    #                 else:
    #                     # 对偏置层进行裁剪
    #                     input_size = v.size(0)
    #                     if 'fc' in k:
    #                         input_idx_i_m = torch.arange(input_size, device=v.device)
    #                         idx[m][k] = input_idx_i_m
    #                     else:
    #                         input_idx_i_m = idx_i[m]
    #                         idx[m][k] = input_idx_i_m
    #             else:
    #                 pass  # 非权重或偏置层，跳过
    #
    #     return idx  # 返回每个客户端的裁剪索引


    def split_model(self, user_idx):  # 为每个客户端分配不同的全局子模型
        idx_i = [None for _ in range(len(user_idx))]
        idx = [OrderedDict() for _ in range(len(user_idx))]
        p=0
        device = torch.device("cuda:0")
        #???第一个问题:每个客户端的裁剪量不同导致聚合后，每个客户端的模型参数不同，故再下一次进行聚合时，是不是需要取各自的模型参数，
        for k, v in self.global_parameters.items():
            p+=1
            parameter_type = k.split('.')[-1]
            for m in range(len(user_idx)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            if 'conv1' in k or 'conv2' in k :
                                if idx_i[m] is None:
                                    idx_i[m] = torch.arange(input_size, device=v.device)
                                input_idx_i_m = idx_i[m]
                                scaler_rate = self.model_rate[user_idx[m]]
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]
                                idx_i[m] = output_idx_i_m
                            elif 'shortcut' in k :
                                input_idx_i_m = idx[m][k.replace('shortcut', 'conv1')][1]
                                output_idx_i_m = idx_i[m]
                            elif 'fc' in k :
                                input_idx_i_m = idx_i[m]
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                            else:
                                pass
                                # raise ValueError('Not valid k')
                            idx[m][k] = (output_idx_i_m, input_idx_i_m)


                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                    else:
                        input_size = v.size(0)
                        if 'fc' in k :
                            input_idx_i_m = torch.arange(input_size, device=v.device)
                            idx[m][k] = input_idx_i_m
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                else:
                    pass
        return idx  # 记录了客户端的以字典形式记录了输出输出通道的索引



    def global_model(self, user_idx):  # 为每个客户端分配不同的全局子模型
        idx_i = [None for _ in range(len(user_idx))]
        idx = [OrderedDict() for _ in range(len(user_idx))]
        p=0
        for k, v in self.global_parameters.items():
            p+=1
            parameter_type = k.split('.')[-1]
            for m in range(len(user_idx)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            if 'conv1' in k or 'conv2' in k :
                                if idx_i[m] is None:
                                    idx_i[m] = torch.arange(input_size, device=v.device)
                                input_idx_i_m = idx_i[m]
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                                idx_i[m] = output_idx_i_m
                            # elif 'downsample' in k :
                            #     input_idx_i_m = idx[m][k.replace('downsample', 'conv1')][1]
                            #     output_idx_i_m = idx_i[m]
                            elif 'fc' in k :
                                input_idx_i_m = idx_i[m]
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                            else:
                                pass
                                # raise ValueError('Not valid k')
                            idx[m][k] = (output_idx_i_m, input_idx_i_m)


                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                    else:
                        input_size = v.size(0)
                        if 'fc' in k :
                            input_idx_i_m = torch.arange(input_size, device=v.device)
                            idx[m][k] = input_idx_i_m
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                else:
                    pass
        return idx  # 记录了90个客户端的以字典形式记录了输出输出通道的索引


    def distribute(self, user_idx,clients):#分離

        param_idx = self.split_model(user_idx)
        local_parameters = [OrderedDict() for _ in range(len(user_idx))]
        for client_idx, client in enumerate(clients):
            client_pramer=self.get_model_params_with_metadata(client.model)
            for k, v in client_pramer.items():
                    parameter_type = k.split('.')[-1]
                # for m in range(len(user_idx)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if 'weight' in parameter_type:
                            if v.dim() > 1:
                                if 'downsample' in k:
                                    pass
                                    # 假设 downsample 是卷积层，按 param_idx 索引切分
                                    # local_parameters[m][k] = copy.deepcopy(v[torch.meshgrid(param_idx[m][k])])
                                else:
                                    #对于其他卷积层，按 param_idx 索引切分
                                    local_parameters[ client_idx][k] = copy.deepcopy(v[torch.meshgrid(param_idx[ client_idx][k])])
                            else:
                                local_parameters[ client_idx][k] = copy.deepcopy(v[param_idx[ client_idx][k]])
                        else:
                            local_parameters[ client_idx][k] = copy.deepcopy(v[param_idx[ client_idx][k]])
                    else:
                        local_parameters[ client_idx][k] = copy.deepcopy(v)
        return local_parameters, param_idx



# import copy
# import torch
# import numpy as np
# from collections import OrderedDict
#
# class Federation:
#
#     def __init__(self, global_parameters, rate):
#         self.global_parameters = global_parameters
#         self.rate = rate
#         self.model_rate = rate
#
#     def split_model(self, user_idx):  # 为每个客户端分配不同的全局子模型
#         idx_i = [None for _ in range(len(user_idx))]
#         idx = [OrderedDict() for _ in range(len(user_idx))]
#         p = 0
#
#         for k, v in self.global_parameters.items():
#             p += 1
#             parameter_type = k.split('.')[-1]
#             for m in range(len(user_idx)):
#                 if 'weight' in parameter_type or 'bias' in parameter_type:
#                     if parameter_type == 'weight':
#                         if v.dim() > 1:  # 对于多维权重（卷积层、全连接层等）
#                             input_size = v.size(1)
#                             output_size = v.size(0)
#
#                             # 处理卷积层（conv1 和 conv2）
#                             if 'conv1' in k or 'conv2' in k:
#                                 if idx_i[m] is None:
#                                     idx_i[m] = torch.arange(input_size, device=v.device)
#                                 input_idx_i_m = idx_i[m]
#                                 scaler_rate = self.model_rate[user_idx[m]]
#                                 local_output_size = int(np.ceil(output_size * scaler_rate))
#                                 output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]
#                                 idx_i[m] = output_idx_i_m
#                                 idx[m][k] = (output_idx_i_m, input_idx_i_m)
#
#                             # 处理 downsample 层（shortcut）
#                             elif 'downsample' in k:
#                                 # 直接查找对应的 conv1 或者其它卷积层（不只是替换名称）
#                                 if 'conv' in k:  # 例如 "downsample.0" 对应 "conv1"
#                                     input_idx_i_m = idx[m].get(k.replace('downsample', 'conv1'), (None, None))[1]
#                                 else:
#                                     # 对于其它的 downsample 层，根据实际情况处理
#                                     input_idx_i_m = idx[m].get(k.replace('downsample', 'conv1'), (None, None))[1]
#                                 output_idx_i_m = idx_i[m]
#                                 idx[m][k] = (output_idx_i_m, input_idx_i_m)
#
#                             # 处理全连接层（fc）
#                             elif 'fc' in k:
#                                 input_idx_i_m = idx_i[m]
#                                 output_idx_i_m = torch.arange(output_size, device=v.device)
#                                 idx[m][k] = (output_idx_i_m, input_idx_i_m)
#                         else:  # 对于一维权重（通常是 fc 层的权重）
#                             input_idx_i_m = idx_i[m]
#                             idx[m][k] = input_idx_i_m
#                     else:
#                         # 对偏置层进行裁剪
#                         input_size = v.size(0)
#                         if 'fc' in k:
#                             input_idx_i_m = torch.arange(input_size, device=v.device)
#                             idx[m][k] = input_idx_i_m
#                         else:
#                             input_idx_i_m = idx_i[m]
#                             idx[m][k] = input_idx_i_m
#                 else:
#                     pass  # 非权重或偏置层，跳过
#
#         return idx  # 返回每个客户端的裁剪索引
#
#     def distribute(self, user_idx,clients):#分離
#
#         param_idx = self.split_model(user_idx)
#         local_parameters = [OrderedDict() for _ in range(len(user_idx))]
#         for client_idx, client in enumerate(clients):
#             for k, v in client.model.state_dict().items():
#                 parameter_type = k.split('.')[-1]
#                 for m in range(len(user_idx)):
#                     if 'weight' in parameter_type or 'bias' in parameter_type:
#                         if 'weight' in parameter_type:
#                             if v.dim() > 1:
#                                 # if 'downsample' in k:
#                                 #     pass
#                                 #     # 假设 downsample 是卷积层，按 param_idx 索引切分
#                                 #     # local_parameters[m][k] = copy.deepcopy(v[torch.meshgrid(param_idx[m][k])])
#                                 # else:
#                                     # 对于其他卷积层，按 param_idx 索引切分
#                                     local_parameters[m][k] = copy.deepcopy(v[torch.meshgrid(param_idx[m][k])])
#                             else:
#                                 local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])
#                         else:
#                             local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])
#                     else:
#                         local_parameters[m][k] = copy.deepcopy(v)
#             return local_parameters, param_idx
#
#


    # def distribute(self, user_idx, clients):  # 分发裁剪后的参数
    #     param_idx = self.split_model(user_idx)
    #     local_parameters = [OrderedDict() for _ in range(len(user_idx))]
    #
    #     for client_idx, client in enumerate(clients):
    #         for k, v in client.model.state_dict().items():
    #             parameter_type = k.split('.')[-1]
    #             for m in range(len(user_idx)):
    #                 if 'weight' in parameter_type or 'bias' in parameter_type:
    #                     if 'weight' in parameter_type:
    #                         if v.dim() > 1:
    #                             # 对卷积层进行裁剪（按照 param_idx 索引）
    #                             local_parameters[m][k] = copy.deepcopy(torch.tensor(v[torch.meshgrid(param_idx[m][k])]))
    #                         else:
    #                             # 对于一维权重（例如 fc 层的权重）
    #                             local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])
    #                     else:
    #                         # 对偏置层进行裁剪
    #                         local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])
    #                 else:
    #                     # 其他参数（非权重和偏置）直接分发
    #                     local_parameters[m][k] = copy.deepcopy(v)
    #
    #     return local_parameters, param_idx  # 返回裁剪后的本地参数和索引
    #
