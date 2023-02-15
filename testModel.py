import torch
import numpy as np

# Validation

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_model(dataloader,network,lossfn):
	global best_acc
	network.eval()
	test_loss = 0
	correct = 0
	total = 0
	batches = 0
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(dataloader):
			#inputs = list(inputs.values())
			#inputs = inputs[0]

			inputs, targets = inputs.to(device), targets.to(device)
			outputs = network(inputs)
			loss = lossfn(outputs, targets)

			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
			batches +=1

			#print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			#             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
	print(len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
				 % (test_loss/(batches), 100.*correct/total, correct, total))
