#Training

def train_model(dataloader,network,lossfn,optimizer):
    network.train()
    train_loss = 0
    correct = 0
    total = 0
    batches = 0
	batch_idx = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = list(inputs.values())
        inputs = inputs[0]
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = network(np.transpose(inputs,(0,3,1,2)))
        loss = lossfn(outputs, targets)
        loss.backward()
        optimizer.step()
        batches +=1

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(batches, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batches), 100.*correct/total, correct, total))