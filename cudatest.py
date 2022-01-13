import torch
import torch.nn as nn
import timm


class myModel1(nn.Module):
    def __init__(self):
        super(myModel1, self).__init__()
        self.encoder1 = timm.create_model('resnet50', pretrained=True, num_classes=0)
        self.encoder2 = timm.create_model('resnet50', pretrained=True, num_classes=0)
        self.head = nn.Sequential(
            nn.Linear(2048*2, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 128)
            )

    def forward(self, data1, data2):
        feature1 = self.encoder1(data1)
        feature2 = self.encoder2(data2)
        # feature = torch.flatten(feature, 1)
        feature1 = torch.flatten(feature1, 1)
        feature2 = torch.flatten(feature2, 1)
        feature = torch.nn.functional.normalize(torch.cat((feature1, feature2), 1), dim=1)
        return feature

class twobranch(nn.Module):
    def __init__(self):
        super(twobranch, self).__init__()
        self.encoder = myModel1()
        if torch.cuda.device_count() > 1:
            self.encoder.encoder1 = torch.nn.DataParallel(self.encoder.encoder1)
            self.encoder.encoder2 = torch.nn.DataParallel(self.encoder.encoder2)
        best_model_path = '/mnt/caizy/SupContrast/save/SupCon/cifar10_models/SupCon_cifar10_resnet50_lr_0.002_decay_0.0001_bsz_8_temp_0.05_trial_0_0923_2b_128_cosine/last.pth'
        para_state_dict = torch.load(best_model_path)
        # temp.load_state_dict(para_state_dict['model'].module)
        self.encoder.load_state_dict(para_state_dict['model'])
        self.encoder.head = nn.Linear(4096,3)
        # self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        # print(self.encoder)
        # self.decision = nn.Linear(128, 3)

    def forward(self, fundus_img, thick_img):
        feature1 = self.encoder(fundus_img, thick_img)
        # feature1 = torch.flatten(feature1, 1)
        # feature2 = torch.flatten(feature2, 1)
        print(feature1.shape)
        return feature1


opt = None
A = torch.rand(1, 3,1024,1024)
B = torch.rand(1, 3,384,384)
model = twobranch().to(0)
model.eval()
C:torch.FloatTensor=model(A,B)