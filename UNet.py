import torch.nn as nn
import torch


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, X):
        return self.block(X)


class UNet(nn.Module):
    def __init__(self, input_channels, out_channels, **kwargs):
        super(UNet, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.conv1_1 = ConvBlock(input_channels, 64, 3)
        self.conv1_2 = ConvBlock(64, 64, 3)
        self.down_sample1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_1 = ConvBlock(64, 128, 3)
        self.conv2_2 = ConvBlock(128, 128, 3)
        self.down_sample2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_1 = ConvBlock(128, 256, 3)
        self.conv3_2 = ConvBlock(256, 256, 3)
        self.down_sample3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4_1 = ConvBlock(256, 512, 3)
        self.conv4_2 = ConvBlock(512, 512, 3)
        self.down_sample4 = nn.MaxPool2d(kernel_size=5, stride=5)
        self.conv5_1 = ConvBlock(512, 1024, 3)
        self.conv5_2 = ConvBlock(1024, 1024, 3)
        self.up_sample1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=5, stride=5, padding=0)
        self.conv6_1 = ConvBlock(1024, 512, 3)
        self.conv6_2 = ConvBlock(512, 512, 3)
        self.up_sample2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0)
        self.conv7_1 = ConvBlock(512, 256, 3)
        self.conv7_2 = ConvBlock(256, 256, 3)
        self.up_sample3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0)
        self.conv8_1 = ConvBlock(256, 128, 3)
        self.conv8_2 = ConvBlock(128, 128, 3)
        self.up_sample4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.conv9_1 = ConvBlock(128, 64, 3)
        self.conv9_2 = ConvBlock(64, out_channels, 3)

        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def forward(self, X, testing=False):
        batch_size = X.shape[0]
        X1 = self.conv1_2(self.conv1_1(X))
        X1_down = self.down_sample1(X1)
        X2 = self.conv2_2(self.conv2_1(X1_down))
        X2_down = self.down_sample2(X2)
        X3 = self.conv3_2(self.conv3_1(X2_down))
        X3_down = self.down_sample3(X3)
        X4 = self.conv4_2(self.conv4_1(X3_down))
        X4_down = self.down_sample4(X4)
        X5 = self.conv5_2(self.conv5_1(X4_down))
        X6 = self.up_sample1(X5)
        X6_up = torch.cat((X6, X4), dim=1)
        X7 = self.up_sample2(self.conv6_2(self.conv6_1(X6_up)))
        X7_up = torch.cat((X7, X3), dim=1)
        X8 = self.up_sample3(self.conv7_2(self.conv7_1(X7_up)))
        X8_up = torch.cat((X8, X2), dim=1)
        X9 = self.up_sample4(self.conv8_2(self.conv8_1(X8_up)))
        X9_up = torch.cat((X9, X1), dim=1)
        out = self.conv9_2(self.conv9_1(X9_up))
        '''
        print(X.shape)
        print(X1.shape, X1_down.shape)
        print(X2.shape, X2_down.shape)
        print(X3.shape, X3_down.shape)
        print(X4.shape, X4_down.shape)
        print(X5.shape)
        print(X6.shape, X6_up.shape)
        print(X7.shape, X7_up.shape)
        print(X8.shape, X8_up.shape)
        print(X9.shape, X9_up.shape)
        '''
        '''
        out = out.reshape(batch_size, self.out_channels, -1)
        if testing:
            out = self.softmax(out)
        '''
        return out

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


if __name__ == '__main__':
    device = 'cpu'
    model = UNet(9, 256).to(device)
    inp = torch.rand(1, 9, 360, 640)
    out = model(inp)
    print('out = {}'.format(out.shape))
