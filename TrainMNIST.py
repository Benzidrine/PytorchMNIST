from NeuralNetwork import MNISTModel
import torch

mnistModel = MNISTModel()
try:
    mnistModel.load_model_checkpoint()
except:
    print("Model failed to load") 
print(mnistModel)
if torch.cuda.is_available():  
    mnistModel = mnistModel.to(torch.device("cuda:0"))
mnistModel.train(20)
mnistModel.save_model_checkpoint()
mnistModel.test_inference()
