import os
import torch
from tqdm import tqdm

from model_architecture import mnist_model
from dataset import mnist_dataset


if __name__ == "__main__":
    model_dir = "weights/mnist/"
    model_name = "mnist_model"
    display_freq = 100
    num_epoch = 5

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, model_name)

    dataset = mnist_dataset(training=True)
    dataloader = torch.utils.data.DataLoader(dataset=dataset.train_data, batch_size=64, shuffle=True)

    net = mnist_model()
    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    print(net)

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    net.to(device)

    for epoch_idx in range(num_epoch):
        print("EPOCH %d" % (epoch_idx+1))
        batch_loss = 0.0
        batch_correct = 0.0
        epoch_loss = 0.0
        epoch_correct = 0
        for batch_idx, data in enumerate(tqdm(dataloader), 1):
            x = data[0]  # data["input"]
            y = data[1]  # data["label"]

            try:
                x, y = x.to(device, dtype=torch.float32), y.to(device)

                optimizer.zero_grad()

                output = net(x)
                pred = output.max(1, keepdim=True)[1]

                loss = torch.nn.CrossEntropyLoss()(output, y)

                loss.backward()
                optimizer.step()

                batch_loss += loss.item()
                batch_correct += pred.eq(y.view_as(pred)).sum().item()

                epoch_loss += loss.item()
                epoch_correct += pred.eq(y.view_as(pred)).sum().item()

                if batch_idx % display_freq == 0:
                    print("Batch %d, Training Loss: %.4f, Training ACC: %.4f" % (
                        batch_idx,
                        batch_loss / (dataloader.batch_size * display_freq),
                        100 * batch_correct / (dataloader.batch_size * display_freq)))
                    batch_loss = 0.0
                    batch_correct = 0.0

            except Exception as e:
                print(e)
                pass

    torch.save(net, model_path)

