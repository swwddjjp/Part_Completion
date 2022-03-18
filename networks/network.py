from sqlalchemy import false
import torch
import torch.nn as nn



class Generator_fc(nn.Module):
    def __init__(self,n_features=(64,128,512,1024),latent_dim=32,output_pts=512,bn=False):
        super(Generator_fc,self).__init__()
        self.n_features = list(n_features)
        self.output_pts = output_pts
        self.latent_dim = latent_dim

        model = []
        prev_nf = self.latent_dim
        for idx, nf in enumerate(self.n_features):
            fc_layer = nn.Linear(prev_nf, nf)
            model.append(fc_layer)

            if bn:
                bn_layer = nn.BatchNorm1d(nf)
                model.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf

        fc_layer = nn.Linear(self.n_features[-1], output_pts*3)
        model.append(fc_layer)

        self.model = nn.Sequential(*model)

    
    def forward(self, x):
        x = self.model(x)
        x = x.view((-1, 3, self.output_pts))
        return x



class Discriminator(nn.Module):
    def __init__(self, n_filters=(64, 128, 256, 256),n_features=(128,64),latent_dim=512,output=1,bn=False):
        super(Discriminator,self).__init__()
        self.n_filters = list(n_filters) + [latent_dim]

        self.n_features = list(n_features)
        self.output_pts = output
        self.latent_dim = latent_dim

        model = []

        prev_nf = 3
        for idx, nf in enumerate(self.n_filters):
            conv_layer = nn.Conv1d(prev_nf, nf, kernel_size=1, stride=1)
            model.append(conv_layer)

            if bn:
                bn_layer = nn.BatchNorm1d(nf)
                model.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf

        prev_nf = self.latent_dim
        for idx, nf in enumerate(self.n_features):
            fc_layer = nn.Linear(prev_nf, nf)
            model.append(fc_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf

        fc_layer = nn.Linear(self.n_features[-1], output)
        model.append(fc_layer)

        self.model = nn.Sequential(*model)

    def forward(self,x):
        validity = self.model(x)
        return validity
