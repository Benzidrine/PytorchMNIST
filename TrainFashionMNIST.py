import torch
from FashionMNIST import FashionMNISTModel

mnistModel = FashionMNISTModel()
try:
    mnistModel.load_model_checkpoint()
except:
    print("Model failed to load") 
if torch.cuda.is_available():  
    mnistModel = mnistModel.to(torch.device("cuda:0"))
print("Model:",mnistModel)
mnistModel.train(1)
mnistModel.save_model_checkpoint()
mnistModel.test_inference()