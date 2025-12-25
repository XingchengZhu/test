import copy
import logging
import torch
from torch import nn

from convs.cifar_resnet import resnet32
from convs.resnet import resnet18, resnet34, resnet50
from convs.ucir_cifar_resnet import resnet32 as cosine_resnet32
from convs.ucir_resnet import resnet18 as cosine_resnet18
from convs.ucir_resnet import resnet34 as cosine_resnet34
from convs.ucir_resnet import resnet50 as cosine_resnet50
from convs.linears import SimpleLinear
from convs.modified_represnet import resnet18_rep, resnet34_rep
from convs.resnet_cbam import resnet18_cbam, resnet34_cbam, resnet50_cbam

import torch
from torch import nn
import torch.nn.functional as F

class CombinedCVAE(nn.Module):
    def __init__(self, feature_dim, num_classes, latent_dim=128, hidden_dims=[512, 256], dropout_prob=0.1):
        super(CombinedCVAE, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        input_dim = feature_dim + num_classes

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(inplace=True)
        )
        self.fc_mu = nn.Linear(hidden_dims[1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[1], latent_dim)

        decoder_input_dim = latent_dim + num_classes
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[0], feature_dim)
        )

    def encode(self, x, labels):
        """
        x: [batch_size, feature_dim]
        labels: [batch_size, num_classes]
        """
        inputs = torch.cat([x, labels], dim=1)
        h = self.encoder(inputs)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        inputs = torch.cat([z, labels], dim=1)
        return self.decoder(inputs)

    def forward(self, x, labels):
        """
        x: [B, feature_dim]
        labels: [B, num_classes]
        """
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, labels)
        return recon_x, mu, logvar

    def generate(self, labels, device):
        B = labels.size(0)
        z = torch.randn(B, self.latent_dim).to(device)
        return self.decode(z, labels)

def get_convnet(args, pretrained=False):
    name = args["convnet_type"].lower()
    if name == "resnet32":
        return resnet32()
    elif name == "resnet18":
        return resnet18(pretrained=pretrained, args=args)
    elif name == "resnet34":
        return resnet34(pretrained=pretrained, args=args)
    elif name == "resnet50":
        return resnet50(pretrained=pretrained, args=args)
    elif name == "cosine_resnet18":
        return cosine_resnet18(pretrained=pretrained, args=args)
    elif name == "cosine_resnet32":
        return cosine_resnet32()
    elif name == "cosine_resnet34":
        return cosine_resnet34(pretrained=pretrained, args=args)
    elif name == "cosine_resnet50":
        return cosine_resnet50(pretrained=pretrained, args=args)
    elif name == "resnet18_rep":
        return resnet18_rep(pretrained=pretrained, args=args)
    elif name == "resnet18_cbam":
        return resnet18_cbam(pretrained=pretrained, args=args)
    elif name == "resnet34_cbam":
        return resnet34_cbam(pretrained=pretrained, args=args)
    elif name == "resnet50_cbam":
        return resnet50_cbam(pretrained=pretrained, args=args)
    else:
        raise NotImplementedError("Unknown type {}".format(name))

class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()
        self.convnet = get_convnet(args, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
    
        out.update(x)
        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self



class CGRIncrementalNet(BaseNet):
    
    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam
        if self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        return SimpleLinear(in_dim, out_dim)

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        out.update(x)
        if hasattr(self, "gradcam") and self.gradcam:
            out["gradcam_gradients"] = self._gradcam_gradients
            out["gradcam_activations"] = self._gradcam_activations
        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        if hasattr(self.convnet, "last_conv"):
            self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(backward_hook)
            self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(forward_hook)

class CGRNet(CGRIncrementalNet):

    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained, gradcam)
        self.args = args
        self.transfer = SimpleLinear(self.feature_dim, self.feature_dim)
        self.total_classes = args.get("total_classes", 100) * 4 + 50
        self.cvae = CombinedCVAE(
            feature_dim=self.feature_dim,
            num_classes=self.total_classes,
            latent_dim=128,
            hidden_dims=[512, 256],
            dropout_prob=0.1
        )

    def update_fc(self, num_old, num_total, num_aux):
        fc = SimpleLinear(self.feature_dim, num_total+num_aux)
        if self.fc is not None:
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:num_old] = weight[:num_old]
            fc.bias.data[:num_old] = bias[:num_old]
        del self.fc
        self.fc = fc

        transfer = SimpleLinear(self.feature_dim, self.feature_dim)
        transfer.weight = nn.Parameter(torch.eye(self.feature_dim))
        transfer.bias = nn.Parameter(torch.zeros(self.feature_dim))
        del self.transfer
        self.transfer = transfer
