from collections import OrderedDict
import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class Encoder(nn.Module):
    def __init__(self, conv3 = None, conv4 = None, fc1 = None, fc2 = None):
        super(Encoder, self).__init__()
        # Inputs 3x64x64

        # store as class member to be able to share with other Encoder
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)                           # 32x32x32
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)                          # 64x16x16
        self.conv3 = conv3 if conv3 else nn.Conv2d(64, 128, 3, stride=2, padding=1)     # 128x8x8
        self.conv4 = conv4 if conv4 else nn.Conv2d(128, 256, 3, stride=2, padding=1)    # 256x4x4
        self.fc1 = fc1 if fc1 else nn.Linear(256 * 4 * 4, 1024)                         # 1x1x1024
        self.fc2 = fc2 if fc2 else nn.Linear(1024, 1024)                                # 1x1x1024

        self.nn = nn.Sequential(OrderedDict([
            ('conv1', self.conv1),
            ('conv2', self.conv2),
            ('conv3', self.conv3),
            ('conv4', self.conv4),
            ('flatten', Flatten()),
            ('fc1', self.fc1),
            ('fc2', self.fc2)
        ]))

    def forward(self, x):
        x = self.nn(x)
        return x


class Decoder(nn.Module):
    def __init__(self, deconv1 = None, deconv2 = None):
        super(Decoder, self).__init__()
        # Inputs 1x1x1024
        self.fc1 = nn.Linear(1024, 1024*2*2)  # 256x4x4=1024x2x2
        self.deconv1 = deconv1 if deconv1 is not None else nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1)  # 512x4x4
        self.deconv2 = deconv2 if deconv2 is not None else nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1)   # 256x8x8
        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1)     # 128x16x16
        self.deconv4 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1)      # 64x32x32
        self.deconv5 = nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1)        # 3x64x64

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.fc1(x)
        x = x.view((batch_size, 1024, 2, 2))
        x = self.deconv1(x, output_size=(batch_size, 512, 4, 4))
        x = self.deconv2(x, output_size=(batch_size, 256, 8, 8))
        x = self.deconv3(x, output_size=(batch_size, 128, 16, 16))
        x = self.deconv4(x, output_size=(batch_size, 64, 32, 32))
        x = self.deconv5(x, output_size=(batch_size, 3, 64, 64))
        return x


class AutoEncoder(nn.Module):
    def __init__(self, enc, dec):
        super(AutoEncoder, self).__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x


def build_model():
    domain1_enc = Encoder()
    domain2_enc = Encoder(domain1_enc.conv3, domain1_enc.conv4, domain1_enc.fc1, domain1_enc.fc2)

    domain1_dec = Decoder()
    domain2_dec = Decoder(domain1_dec.deconv1, domain1_dec.deconv2)

    domain1_model = AutoEncoder(domain1_enc, domain1_dec)
    domain2_model = AutoEncoder(domain2_enc, domain2_dec)
    return domain1_model, domain2_model
