import timm
from pprint import pprint
from torchstat import stat
from torchsummary import summary

# model_names = timm.list_models(pretrained=True)
# pprint(model_names)

# m = timm.create_model('vit_tiny_patch16_224', pretrained=False)
m = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False)
summary(m.to(0), input_size=(3, 224, 224), batch_size=-1)
# print(m)