import torch
import numpy as np

#Training
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_model(dataloader,network,lossfn,optimizer,scheduler=None):
	network.train()
	train_loss = 0
	correct = 0
	total = 0
	batches = 0

	for batch_idx, (inputs, targets) in enumerate(dataloader):
		#inputs = list(inputs.values())
		#inputs = inputs[0]
		inputs, targets = inputs.to(device), targets.to(device)
		optimizer.zero_grad()
		outputs = network(inputs)
		loss = lossfn(outputs, targets)
		loss.backward()
		optimizer.step()
		if scheduler != None:
                 scheduler.step()
		batches +=1
		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()

	print(batches, len(dataloader), 'Training:: Loss: %.3f | Acc: %.3f%% (%d/%d)'
					 % (train_loss/(batches), 100.*correct/total, correct, total))
