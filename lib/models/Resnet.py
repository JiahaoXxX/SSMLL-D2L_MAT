
from torchvision import models
import torch.nn as nn

class QuadClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, state_dict=None):
        super(QuadClassifier, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)
        self.fc3 = nn.Linear(in_dim, out_dim)
        self.fc4 = nn.Linear(in_dim, out_dim)

        if state_dict is not None:
            self.fc1.load_state_dict(state_dict[0])
            self.fc2.load_state_dict(state_dict[1])
            self.fc3.load_state_dict(state_dict[2])
            self.fc4.load_state_dict(state_dict[3])

    def transfer_weight(self):
        print("fc3 is loadding the weight of fc1...")
        self.fc3.load_state_dict(self.fc1.state_dict())
        print("fc4 is loadding the weight of fc2...")
        self.fc4.load_state_dict(self.fc2.state_dict())

    def forward(self, x):
        logit1 = self.fc1(x)
        logit2 = self.fc2(x)
        logit3 = self.fc3(x)
        logit4 = self.fc4(x)
        return logit1, logit2, logit3, logit4
 


def create_model(mo, n_classes):

    model = resnet(mo.lower(), n_classes)

    return model

def resnet(mo, nc, pretrain=True):
    
    if 'resnet18' in mo:
        model = models.resnet18(pretrained=pretrain)
    elif 'resnet34' in mo:
        model = models.resnet34(pretrained=pretrain)
    elif 'resnet50' in mo:
        model = models.resnet50(pretrained=pretrain)
    
    if mo in ['resnet18','resnet34', 'resnet50']:
        model.fc = nn.Linear(model.fc.in_features, nc)
    elif mo in ['resnet18_qc','resnet34_qc', 'resnet50_qc']:
        model.fc = QuadClassifier(model.fc.in_features, nc)
        
    return model
