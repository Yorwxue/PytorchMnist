import os
import torch
from tqdm import tqdm

from model_architecture import mnist_model
from dataset import mnist_dataset


if __name__ == "__main__":
    model_dir = "weights/mnist/"
    model_name = "mnist_model"
    model_path = os.path.join(model_dir, model_name)
    display_freq = 100

    if not os.path.exists(model_path):
        print("weighting file not found")
        exit()

    dataset = mnist_dataset(training=False)
    dataloader = torch.utils.data.DataLoader(dataset=dataset.test_data, batch_size=64, shuffle=True)

    net = torch.load(model_path)
    print(net)

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    net.to(device)

    total_loss = 0.
    total_correct = 0.
    batch_loss = 0.0
    batch_correct = 0.0
    for batch_idx, data in enumerate(tqdm(dataloader), 1):
        x = data[0]  # data["input"]
        y = data[1]  # data["label"]

        try:
            x, y = x.to(device, dtype=torch.float32), y.to(device)


            output = net(x)
            pred = output.max(1, keepdim=True)[1]

            loss = torch.nn.CrossEntropyLoss()(output, y)

            batch_loss += loss.item()
            batch_correct += pred.eq(y.view_as(pred)).sum().item()
            total_loss += loss.item()
            total_correct += pred.eq(y.view_as(pred)).sum().item()

            if batch_idx % display_freq == 0:
                print("Batch %d, Testing Loss: %.4f, Testing ACC: %.4f" % (
                    batch_idx,
                    batch_loss / (dataloader.batch_size * display_freq),
                    100 * batch_correct / (dataloader.batch_size * display_freq)))

                batch_loss = 0.0
                batch_correct = 0.0

        except Exception as e:
            print(e)
            pass

    print("Testing Loss: %.4f, Testing ACC: %.4f" % (
        total_loss / len(dataset.test_data),
        100 * total_correct / len(dataset.test_data)))
