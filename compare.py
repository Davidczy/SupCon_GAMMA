import timm
from timm.models import res2net
import torch.nn as nn
import torch
from networks.resnet_big import SupConResNet
from main_supcon import myModel


# class myModel(nn.Module):
#     def __init__(self):
#         super(myModel, self).__init__()
#         self.encoder = timm.create_model('resnet50', pretrained=True, num_classes=0)
#         self.head = nn.Sequential(
#             nn.Linear(2048, 2048),
#             nn.ReLU(inplace=True),
#             nn.Linear(2048, 128)
#         )

#     def forward(self, data):
#         feature = self.encoder(data)
#         # feature = self.head(feature.squeeze())
#         # feature = self.head(feature)
#         # feature = torch.nn.functional.normalize(feature, dim=1)
#         feature = torch.flatten(feature, 1)
#         print(feature.shape)
#         feature = torch.nn.functional.normalize(self.head(feature), dim=1)
#         return feature

A = myModel(opt=None)
A.encoder = nn.DataParallel(A.encoder).cuda()
# B = SupConResNet('resnet50')

# print(A.encoder.global_pool)
# A.encoder.global_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
best_model_path = '/mnt/caizy/SupContrast/save/SupCon/cifar10_models/SupCon_cifar10_resnet50_lr_0.001_decay_0.0001_bsz_8_temp_0.01_trial_0_0920_pretrained_cosine/ckpt_epoch_20.pth'
para_state_dict = torch.load(best_model_path)
A.load_state_dict(para_state_dict['model'])
print(A)
A = nn.Sequential(*list(A.children())[:-1])
print(A)
# print(A.encoder.global_pool)
# a = torch.rand(1,3,224,224)
# A.eval()
# B.eval()
# b1 = B(a)
# a1 = A(a)


# print(a1.size())
# print(b1.size())