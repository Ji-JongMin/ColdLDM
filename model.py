import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_ch=1, filter_num=64, layer_num=10, bnorm_flag=True):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(input_ch, filter_num, kernel_size=3, padding=1))
        if bnorm_flag:
            self.layers.append(nn.BatchNorm2d(filter_num))
        self.layers.append(nn.ReLU(inplace=True))

        for _ in range(layer_num - 1):
            self.layers.append(nn.Conv2d(filter_num, filter_num, kernel_size=3, padding=1))
            if bnorm_flag:
                self.layers.append(nn.BatchNorm2d(filter_num))
            self.layers.append(nn.ReLU(inplace=True))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, output_ch=1, filter_num=64, layer_num=10, bnorm_flag=True):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(layer_num - 1):
            self.layers.append(nn.Conv2d(filter_num, filter_num, kernel_size=3, padding=1))
            if bnorm_flag:
                self.layers.append(nn.BatchNorm2d(filter_num))
            self.layers.append(nn.ReLU(inplace=True))

        self.layers.append(nn.Conv2d(filter_num, output_ch, kernel_size=3, padding=1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, input_ch=1, output_ch=1, filter_num=64, encoder_layers=10, decoder_layers=10):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_ch, filter_num, encoder_layers)
        self.decoder = Decoder(output_ch, filter_num, decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)
