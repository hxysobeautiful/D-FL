import argparse, json
import datetime
import os
import logging
import torch, random
from server import *
from client import *
import models, datasets
from model_cv import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
#训练（训练的时候进行裁剪，之后聚合的也是裁剪后的模型参数），聚合，评估

if __name__ == '__main__':
	import torch

	print(torch.cuda.device_count())  # 检查可用的 GPU 数量
	print(torch.cuda.current_device())  # 检查当前使用的设备
	print(torch.cuda.get_device_name(0))  # 检查 GPU 的名称

	# 创建ArgumentParser对象，用于解析命令行参数
	parser = argparse.ArgumentParser(description='Federated Learning')
	# 添加参数，-c或--conf用于指定配置文件
	parser.add_argument('-c', '--conf', dest='conf', default="utils/conf.json")
	# 解析命令行参数
	args = parser.parse_args()
	# print(args.conf)

	with open(args.conf, 'r',encoding="utf-8") as f:
		conf = json.load(f)
	# 例子：conf 没有 'gpu' 键

	device = torch.device("cuda:0")

	# 配置日志记录
	logging.basicConfig(level=logging.DEBUG)  # 设定日志级别
	logger = logging.getLogger()  # 获取根日志记录器
	logger.setLevel(logging.DEBUG)

	# 创建一个FileHandler用于将日志写入文件
	import os
	import time
	import logging

	# 获取当前时间
	ticks = time.gmtime()

	# 日志文件路径
	log_dir = 'experiment_results'  # 目录路径
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)  # 如果目录不存在，创建它

	# 日志文件名
	log_file = f'{log_dir}/{ticks.tm_mon}{ticks.tm_mday}_{ticks.tm_hour}{ticks.tm_min}.log'

	# 设置日志处理器
	file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')  # 设置日志格式
	file_handler.setFormatter(formatter)  # 将格式应用到FileHandler
	logger.addHandler(file_handler)  # 将FileHandler添加到logger

	# 打印模型信息到日志
	logger.info(f"Using device: {device}")
	logger.info(f"Configuration: {conf}")

	# 获取训练和评估数据集
	train_datasets, eval_datasets = datasets.get_dataset(".data/CIFAR10/raw", conf["type"])
	clients = []
	model = models.get_model(conf["model_name"])

	model=model.to(device)
	logger.info(model)

	losses = []
	model=models.get_model(conf["model_name"])
	model.to(device)

	for c in range(conf["no_models"]):
		# 创建 Client 对象，传入配置文件、全局模型的深拷贝、训练数据集和当前模型索引
		clients.append(Client(conf, copy.deepcopy(model), train_datasets, c))


	# 创建Server对象，传入配置文件和评估数据集
	server = Server(conf, eval_datasets,clients,model)
	global_parameters = server.get_model_params_with_metadata(model)

	rate=conf["rate"]

	user_idx = [i for i in range(conf["no_models"])]
	fed = Federation(global_parameters, rate)
	communication_rounds =500
	avg_accuracies = []
	avg_losses = []

	for round in range(communication_rounds):
		logger.info(f"Round {round + 1}/{communication_rounds}")
		print(f"Round {round + 1}/{communication_rounds}")
		num_items = int(len(train_datasets) / conf["no_models"])
		data_split, idx = {}, list(range(len(train_datasets)))
		label_splits = {}
		label = torch.tensor(train_datasets.targets)

		for i in range(conf["no_models"]):
			num_items_i = min(len(idx), num_items)
			data_split[i] = torch.tensor(idx)[torch.randperm(len(idx))[:num_items_i]].tolist()
			label_splits[i] = torch.unique(label[data_split[i]]).tolist()

		# 每个客户端进行本地训练
		client_param_0=[]
		for client in clients:
			client.local_train(client.model,client.train_loader)
			client_param_0.append(server.get_model_params_with_metadata(client.model))


		all_params = [client.get_model_params() for client in clients]

		local_parameters, param_idx = fed.distribute(user_idx,clients)
		global_param_idx = fed.global_model(user_idx)
		# 聚合模型param_idx
		aggregated_params_list = server.aggregate_models(local_parameters, param_idx, clients, global_param_idx,
													   user_idx, label_splits)

		# 将聚合后的模型更新到所有客户端
		for client in clients:
			accc=client.get_model_params()
			client.update_clients(aggregated_params_list, clients)

		avg_accuracy, avg_loss, accuracies, losses = server.evaluate_clients(round, clients)
		for idx, accuracy, loss in zip(user_idx, accuracies, losses):
			logger.info(f"local_test_on_all_clients: client:{idx} th user")
			logger.info(f"{{'training_acc': {accuracy}, 'training_loss': {loss}}}")

