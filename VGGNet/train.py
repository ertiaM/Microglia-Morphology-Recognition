import os
import pprint
import sys
import json

import torch
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms, datasets
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from vgg import VGG16
from model import vgg16


def main():
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.Resize(256),
                                     transforms.RandomCrop((225, 255)),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.RandomVerticalFlip(p=0.5),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.0268, 0.2314, 0.0270), (0.0467, 0.0785, 0.0467))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.0268, 0.2314, 0.0270), (0.0467, 0.0785, 0.0467))])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "./dataset", "data")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    total_dataset = datasets.ImageFolder(root=image_path,
                                         transform=data_transform["train"])

    spilt_point1 = int(0.1*len(total_dataset))
    spilt_point2 = int(0.9*len(total_dataset))
    index_list = list(range(len(total_dataset)))
    np.random.shuffle(index_list)
    test_index = index_list[:spilt_point1] + index_list[spilt_point2:]
    train_index = index_list[spilt_point1:spilt_point2]

    tr_sampler = SubsetRandomSampler(train_index)
    val_sampler = SubsetRandomSampler(test_index)


    flower_list = total_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 8
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 2])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(total_dataset,
                                               batch_size=batch_size, sampler=tr_sampler,
                                               num_workers=nw)

    validate_loader = torch.utils.data.DataLoader(total_dataset,
                                                  batch_size=batch_size, sampler=val_sampler,
                                                  num_workers=nw)


    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()

    model_name = "vgg16"
    net = vgg16(pretrained=True)
    # weight_path = os.path.join(data_root, "\\Projects\\cells\\reference\\Test3_vggnet\\models", "vgg16-397923af.pth")
    # net = pretrained_vgg(weight_path)
    net.to(device)
    pprint.pprint(net)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), 1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-4, amsgrad=True)

    epochs = 30
    best_acc = 0.0
    save_path = './{}Net.pth'.format(model_name)
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            val_num = 0
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_num = val_num + 1
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num * batch_size
        print(val_num)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
