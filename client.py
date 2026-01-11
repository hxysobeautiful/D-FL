
import models, torch, copy
from server import *
import torch.nn.functional as F
import numpy as np
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 只让 PyTorch 看到 GPU 2
class Client(object):

	def __init__(self, conf, model, train_dataset, id ):
		
		# 初始化配置
		self.conf = conf

		self.model= model

		# 初始化客户端ID
		self.client_id = id

		# 初始化训练数据集
		self.train_dataset = train_dataset

		# 获取数据集长度
		all_range = list(range(len(self.train_dataset)))
		# 计算每个模型的数据长度
		data_len = int(len(self.train_dataset) /conf["no_models"])
		# 获取当前模型的数据索引
		train_indices = all_range[id * data_len: (id + 1) * data_len]

		# 获取训练数据加载器
		self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"], 
									sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices))


	def get_model_params(self):
		return [param.data.clone() for param in self.model.parameters()]

	def set_model_params(self, params,client):
		for param, new_param in zip(client.model.parameters(), params):
			a = new_param.float()
			param.data = a.clone()

	def update_clients(self, aggregated_params_list, clients):

		for client, aggregated_params in zip(clients, aggregated_params_list):
			client.set_model_params(aggregated_params,client)


	def local_train(self, model,train_loader):#客户端进行模型本地训练
		optimizer = torch.optim.SGD(self.model.parameters(), lr=self.conf['lr'],
									momentum=self.conf['momentum'])


		device = torch.device("cuda:0")
		# print('device',device)
		self.model.train()
		# 进行本地训练
		for e in range(self.conf["local_epochs"]):
			# print(torch.cuda.current_device())  # 打印当前设备的 ID

			# 遍历训练数据
			for batch_id, batch in enumerate(train_loader):
				data, target = batch

				data=data.to(device)# 将数据移动到GPU上
				# print(data.device)
				target=target.to(device)
				# 梯度清零
				optimizer.zero_grad()

				model=model.to(device)
				# 前向传播
				output = self.model(data)
				# 计算损失
				loss = torch.nn.functional.cross_entropy(output, target).to(device).float()

				# 反向传播
				loss.backward()

				# 更新参数
				optimizer.step()
			# 打印当前训练轮数
			print("Epoch %d done." % e)




