def train_log(path, e_acc, e_loss, lr, epoch):
    with open(path, 'a+') as f:
        e_acc = str(e_acc)
        e_loss = str(e_loss)
        lr = str(lr)
        epoch = str(epoch + 1)
        f.write('Epoch:' + epoch + ' ' + 'Learning rate:' + lr + ' ' +
                'Accuracy:' + e_acc + ' ' + ' Loss:' + e_loss + '\n')


def eval_log(path, e_acc, epoch):
    with open(path, 'a+') as f:
        e_acc = str(e_acc)
        epoch = str(epoch + 1)
        f.write('Epoch:' + epoch + ' ' + 'Accuracy:' + e_acc + ' ' + '\n')