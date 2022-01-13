import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import os


path1 = '/mnt/caizy/SupContrast/train1_feature.npy'
path2 = '/mnt/caizy/SupContrast/train1_idx.npy'
valpath1 = '/mnt/caizy/SupContrast/val1_feature.npy'
valpath2 = '/mnt/caizy/SupContrast/val1_idx.npy'
label_file = '/mnt/caizy/my_classfication/glaucoma_grading_training_GT.xlsx'
data = np.load(path1).squeeze()
index = np.load(path2).squeeze()

valdata = np.load(valpath1).squeeze()
valindex = np.load(valpath2).squeeze()

class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.decision_branch = nn.Sequential(
            nn.Linear(128, 3),
            nn.Softmax()
        )

    def forward(self, data):
        logit = self.decision_branch(data)
        return logit

label = {row['data']: row[1:].values 
                        for _, row in pd.read_excel(label_file).iterrows()}
file_list = [[f, label[f].argmax()] for f in index]
val_filelist = [[f, label[f].argmax()] for f in valindex]
# print(file_list)
dataset = []
val_dataset = []
for i in range(len(file_list)):
    dataset.append([file_list[i][1],data[i]])
for i in range(len(val_filelist)):
    val_dataset.append([val_filelist[i][1],valdata[i]])
# print(type(dataset))
# print(np.array(dataset))
# print(a)
train_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=8,
    shuffle = True,
    num_workers = 1
    # sampler = train_sampler
    )

val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=1,
    shuffle = True,
    num_workers = 1
    # sampler = train_sampler
    )

model = myModel()
model = model.to(0)
optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )
criterion = nn.CrossEntropyLoss()

iter = 0
model.train()
avg_loss_list = []
avg_kappa_list = []
best_kappa = 0.
while iter < 30000:
    for data in train_loader:
        model.train()
        iter += 1
        if iter > 30000:
            break
        # idx = data[0]
        label = data[0].to(0)
        feature = (data[1]*10).to(0)

        optimizer.zero_grad()
        logits = model(feature)
        loss = criterion(logits, label)
        for p, l in zip(logits.cpu().detach().numpy().argmax(1), label.cpu().detach().numpy()):
            avg_kappa_list.append([p, l])
        loss.backward()
        optimizer.step()

        avg_loss_list.append(loss.cpu().detach().numpy())

        if iter % 100 == 0:
            avg_loss = np.array(avg_loss_list).mean()
            avg_kappa_list = np.array(avg_kappa_list)
            avg_kappa = cohen_kappa_score(avg_kappa_list[:, 0], avg_kappa_list[:, 1], weights='quadratic')
            avg_loss_list = []
            avg_kappa_list = []
            print("[TRAIN] iter={}/{} avg_loss={:.4f} avg_kappa={:.4f}".format(iter, 30000, avg_loss, avg_kappa))
        if iter % 200 == 0:
            model.eval()
            avg_loss_list = []
            cache = []
            with torch.no_grad():
                for data in val_loader:
                    label = data[0].to(0)
                    feature = data[1].to(0)
                    logits = model(feature)
                    for p, l in zip(logits.cpu().detach().numpy().argmax(1), label.cpu().detach().numpy()):
                        cache.append([p, l])

                    loss = criterion(logits, label)
                    avg_loss_list.append(loss.cpu().detach().numpy())
            cache = np.array(cache)
            print(cache)
            kappa = cohen_kappa_score(cache[:, 0], cache[:, 1], weights='quadratic')
            avg_loss = np.array(avg_loss_list).mean()
            print("[EVAL] iter={}/{} avg_loss={:.4f} kappa={:.4f}".format(iter, 20000, avg_loss, kappa))
            if kappa > best_kappa:
                best_kappa=kappa
                os.mkdir("{}f1classbest_model_{:.4f}".format(iter, best_kappa))
                torch.save(model.state_dict(),
                        os.path.join("{}f1classbest_model_{:.4f}".format(iter, best_kappa), 'model.pdparams'))
            model.train()

