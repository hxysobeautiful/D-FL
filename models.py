#
# import torch
# from torchvision import models
#
# def get_model(name="vgg16", pretrained=True):
# 	if name == "resnet18":
# 		model = models.resnet18(pretrained=pretrained)
# 	elif name == "resnet50":
# 		model = models.resnet50(pretrained=pretrained)
# 	elif name == "densenet121":
# 		model = models.densenet121(pretrained=pretrained)
# 	elif name == "alexnet":
# 		model = models.alexnet(pretrained=pretrained)
# 	elif name == "vgg16":
# 		model = models.vgg16(pretrained=pretrained)
# 	elif name == "vgg19":
# 		model = models.vgg19(pretrained=pretrained)
# 	elif name == "inception_v3":
# 		model = models.inception_v3(pretrained=pretrained)
# 	elif name == "googlenet":
# 		model = models.googlenet(pretrained=pretrained)
#
# 	if torch.cuda.is_available():
# 		return model.cuda()
# 	else:
# 		return model

	# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# inputs = inputs.to(device)
import torch
from torchvision import models

def get_model(name="vgg16", pretrained=True):
    if name == "resnet18":
       model = models.resnet18(pretrained=pretrained)
    elif name == "resnet34":
       model = models.resnet50(pretrained=pretrained)
    elif name == "densenet121":
       model = models.densenet121(pretrained=pretrained)
    elif name == "alexnet":
       model = models.alexnet(pretrained=pretrained)
    elif name == "vgg16":
       model = models.vgg16(pretrained=pretrained)
    elif name == "vgg19":
       model = models.vgg19(pretrained=pretrained)
    elif name == "inception_v3":
       model = models.inception_v3(pretrained=pretrained)
    elif name == "googlenet":
       model = models.googlenet(pretrained=pretrained)

    # 检查是否有 GPU 可用
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    return model

# # 使用模型时确保输入在正确设备
# model, device = get_model("vgg16", pretrained=True)
# input_data = torch.randn(1, 3, 224, 224).to(device)  # 将输入数据移动到模型所在的设备
#
# # 前向传播
# output = model(input_data)