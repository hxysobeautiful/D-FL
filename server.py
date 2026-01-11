#模型聚合和评估
import torch
import torch.nn.functional as F
from client import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from model_cv import *
import argparse, json
import datetime
import os
import logging
import torch, random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 只让 PyTorch 看到 GPU 2
from client import *
import logging
import models, datasets
from collections import OrderedDict
import models, datasets
class Server(object):
	
	def __init__(self, conf,eval_datasets,clients,model):

		self.conf = conf
		self.clients = clients
		self.model = model

		
		# self.global_model = models.get_model(self.conf["model_name"])
		
		self.eval_loader = torch.utils.data.DataLoader(eval_datasets, batch_size=self.conf["batch_size"], shuffle=True)
	# def get_globalmodel_params(self,model):
	# 	return [param.data.clone() for param in model.parameters()]

	# def get_model_params_with_metadata(self,model):
	# 	params_with_metadata = {}
	#
	# 	for name, param in model.named_parameters():
	# 		# 提取基础信息
	# 		param_info = {
	# 			'data': param.clone().detach(),  # 克隆参数数据
	# 			'requires_grad': param.requires_grad,  # 是否需要梯度
	# 			'shape': param.shape  # 参数形状
	# 		}
	#
	# 		# 尝试获取卷积层等模块的 kernel_size 或其他相关信息
	# 		module_name = '.'.join(name.split('.')[:-1])  # 通过名称推测所属模块名称
	# 		module = dict(model.named_modules()).get(module_name, None)
	#
	# 		# 如果是卷积层或具有 kernel_size 属性的模块，保留 k 值
	# 		if module and hasattr(module, 'kernel_size'):
	# 			param_info['kernel_size'] = module.kernel_size  # 例如 k 值
	# 		elif module and hasattr(module, 'stride'):
	# 			param_info['stride'] = module.stride  # 其他元信息，如步幅
	#
	# 		# 添加到最终的字典中
	# 		params_with_metadata[name] = param_info
	#
	# 	return params_with_metadata

	def get_model_params_with_metadata(self,model):
		params_with_metadata = OrderedDict()  # 使用 OrderedDict 保证顺序

		for name, param in model.named_parameters():
			params_with_metadata[name] = param.clone().detach()  # 直接存储 tensor

		return params_with_metadata
	def aggregate_models(self, local_parameters, param_idx, clients, global_param_idx, user_idx,
						 label_splits):  # 聚合
		# 创建一个有序字典，用于存储参数
		count = OrderedDict()
		aggregated_params_list = []
		for client_idx, client in enumerate(clients):
			aggregated_params = []
			global_parameters=self.get_model_params_with_metadata(client.model)
			for k, v in global_parameters.items():
				parameter_type = k.split('.')[-1]
				count[k] = v.new_zeros(v.size(), dtype=torch.float32)
				tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
				for m in range(len(local_parameters)):
					if 'weight' in parameter_type or 'bias' in parameter_type:
						if parameter_type == 'weight':
							if v.dim() > 1:

								if 'downsample' in k:
									pass
								elif 'fc' in k:
									if client_idx == m:
										label_split = label_splits[user_idx[m]]
										global_param_idx[m][k] = list(global_param_idx[m][k])
										global_param_idx[m][k][0] = global_param_idx[m][k][0][label_split]


										tmp_v[torch.meshgrid(global_param_idx[m][k])] += global_parameters[k][
											label_split]

										count[k][torch.meshgrid(global_param_idx[m][k])] += 1
									else:
										label_split = label_splits[user_idx[m]]
										param_idx[m][k] = list(param_idx[m][k])
										param_idx[m][k][0] = param_idx[m][k][0][label_split]
										tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][
											label_split]
										count[k][torch.meshgrid(param_idx[m][k])] += 1
								else:
									if client_idx == m:
										tmp_v[torch.meshgrid(global_param_idx[m][k])] += global_parameters[k]
										count[k][torch.meshgrid(global_param_idx[m][k])] += 1
									else:

										tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k]
										count[k][torch.meshgrid(param_idx[m][k])] += 1
							else:
								if client_idx == m:
									tmp_v[global_param_idx[m][k]] += global_parameters[k]
									count[k][global_param_idx[m][k]] += 1
								else:
									tmp_v[param_idx[m][k]] += local_parameters[m][k]
									count[k][param_idx[m][k]] += 1
						else:

							if 'downsample' in k:
								pass
							elif 'fc' in k:
								if client_idx == m:
									label_split = label_splits[user_idx[m]]
									global_param_idx[m][k] = global_param_idx[m][k][label_split]
									tmp_v[global_param_idx[m][k]] += global_parameters[k][label_split]
									count[k][global_param_idx[m][k]] += 1
								else:
									label_split = label_splits[user_idx[m]]
									param_idx[m][k] = param_idx[m][k][label_split]
									tmp_v[param_idx[m][k]] += local_parameters[m][k][label_split]
									count[k][param_idx[m][k]] += 1
							else:
								if client_idx == m:
									tmp_v[global_param_idx[m][k]] += global_parameters[k]
									count[k][global_param_idx[m][k]] += 1
								else:

									tmp_v[param_idx[m][k]] += local_parameters[m][k]
									count[k][param_idx[m][k]] += 1
					else:
						if client_idx == m:
							tmp_v += global_parameters[k]
							count[k] += 1

						else:
							tmp_v += local_parameters[m][k]
							count[k] += 1

				tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
				aggregated_params.append(tmp_v)
			aggregated_params_list.append(aggregated_params)

		return aggregated_params_list

	#

	# def model_eval(self):
	# 	# 计算一个客户端模型在验证集上的损失（这里简单使用一个客户端进行评估）
	# 	client = self.clients[0]  # 使用第一个客户端进行评估
	# 	client.model.eval()
	# 	total_loss = 0
	# 	total_samples = 0
	# 	correct_predictions = 0
	# 	with torch.no_grad():
	# 		for batch_id, batch in enumerate(self.eval_loader):
	# 			data, target = batch
	# 			if torch.cuda.is_available():
	# 				data = data.cuda()
	# 				target = target.cuda()
	# 			output = client.model(data)
	# 			loss = F.cross_entropy(output, target, reduction='sum')
	# 			total_loss += loss.item()
	# 			total_samples += len(data)
	# 			# 计算准确率
	# 			_, predicted = torch.max(output, 1)
	# 			correct_predictions += (predicted == target).sum().item()
	#
	# 	accuracy = correct_predictions / total_samples
	# 	return total_loss / total_samples, accuracy
	#



	# def evaluate(self,client):
	# 	client.model.eval()
	# 	total_loss = 0
	# 	total_samples = 0
	# 	correct_predictions = 0
	# 	with torch.no_grad():
	# 		for batch_id, batch in enumerate(self.eval_loader):
	# 			data, target = batch
	# 			if torch.cuda.is_available():
	# 				data = data.cuda()
	# 				target = target.cuda()
	# 			output = client.model(data)
	# 			loss = F.cross_entropy(output, target, reduction='sum')
	# 			total_loss += loss.item()
	# 			total_samples += len(data)
	# 			# 计算准确率
	# 			_, predicted = torch.max(output, 1)
	# 			correct_predictions += (predicted == target).sum().item()
	#
	# 	accuracy = correct_predictions / total_samples
	# 	return total_loss / total_samples, accuracy
	def evaluate(self, model):
		model.eval()
		total_loss = 0
		total_samples = 0
		correct_predictions = 0
		device = torch.device("cuda:0")

		# if torch.cuda.is_available():
		# 	model = model.cuda()
		# model.to(device)

		with torch.no_grad():
			for batch_id, batch in enumerate(self.eval_loader):
				data, target = batch
				# if torch.cuda.is_available():
				# 	data = data.cuda()
				# 	target = target.cuda()
				data=data.to(device).float()
				target=target.to(device)
				# 前向传播
				model=model.to(device)
				output = model(data)

				# 计算损失
				loss = torch.nn.functional.cross_entropy(output, target, reduction='sum').to(device).float()
				total_loss += loss.item()
				total_samples += len(data)

				# 计算准确率
				_, predicted = torch.max(output, 1)
				correct_predictions += (predicted == target).sum().item()

		accuracy = correct_predictions / total_samples
		return total_loss / total_samples, accuracy

	def evaluate_clients(self,round,clients):
		# 评估每个客户端的模型性能
		accuracies = []
		losses=[]


		for idx, client in enumerate(clients):
			loss,accuracy = self.evaluate(client.model)
			accuracies.append(accuracy)
			losses.append(loss)

			print(f"Client {idx} accuracy: {accuracy:.2f}%")
			print(f"Client {idx} loss: {loss:.2f}%")


		avg_accuracy = sum(accuracies) / len(accuracies)
		avg_loss=sum(losses) / len(losses)

		print(f"Average accuracy across all clients: {avg_accuracy:.2f}%")
		print(f"Average loss across all clients: {avg_loss:.2f}%")
		return avg_accuracy,avg_loss,accuracies,losses


