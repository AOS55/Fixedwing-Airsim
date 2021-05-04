import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class RandomDataset(Dataset):
	
	def __init__(self, size, length):
		self.len = length
		self.data = torch.randn(length, size)
		
	def __getitem__(self, index):
		return self.data[index]
	
	def __len__(self):
		return self.len


class Model(nn.Module):
	
	def __init__(self, input_size, output_size):
		super(Model, self).__init__()
		self.fc = nn.Linear(input_size, output_size)

	def forward(self, input):
		output = self.fc(input)
		print('\tIn Model: input_size', input.size(),
			'output size', output.size())
		return output

if __name__=='__main__':
	
	input_size = 5
	output_size = 2
	batch_size = 30
	data_size = 100
	
	print(f'input_size:{input_size}, output_size:{output_size}, batch_size:{batch_size}, data_size:{data_size}')	
	
	rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size), batch_size=batch_size, shuffle=True)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	model = Model(input_size, output_size)
	if torch.cuda.device_count() > 1:
		print("Lets use", torch.cuda.device_count(), "GPUs!")
		model = nn.DataParallel(model)
	model.to(device)
	
	for data in rand_loader:
		input = data.to(device)
		output = model(input)
		print('Outside: input size', input.size(), 'output_size', output.size())


