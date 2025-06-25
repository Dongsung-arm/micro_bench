import torch
from utils import convert_to_tflite

class TinyModel(torch.nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()
 
        self.linear1 = torch.nn.Linear(5,5)
     
    def forward(self,x):
        x = self.linear1(x)
 
        return x

tiny_model = TinyModel()
print(tiny_model)
 
sample_input = (torch.randn(3,5),)
print(sample_input)
edge_model = convert_to_tflite(tiny_model,sample_input)
output = edge_model(*sample_input)
edge_model.export('matmul.tflite')
 
print(output)