import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=(1, 1)),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=(1, 1)),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU())
        self.pool0 = nn.Conv2d(in_channels=32, out_channels=32, stride=2, kernel_size=3, padding=(1, 1)) # 512 -> 256
        self.enc_conv1 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=(1, 1)),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=(1, 1)),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU())
        self.pool1 = nn.Conv2d(in_channels=64, out_channels=64, stride=2, kernel_size=3, padding=(1, 1)) # 256 -> 128
        self.enc_conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=(1, 1)),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=(1, 1)),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU())
        self.pool2 = nn.Conv2d(in_channels=128, out_channels=128, stride=2, kernel_size=3, padding=(1, 1)) # 128 -> 64
        self.enc_conv3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=(1, 1)),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=(1, 1)),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU())
        self.pool3 = nn.Conv2d(in_channels=256, out_channels=256, stride=2, kernel_size=3, padding=(1, 1)) # 64 -> 32
        self.enc_conv4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=(1, 1)),
                                       nn.BatchNorm2d(512),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=(1, 1)),
                                       nn.BatchNorm2d(512),
                                       nn.ReLU())
        self.pool4 = nn.Conv2d(in_channels=512, out_channels=512, stride=2, kernel_size=3, padding=(1, 1)) # 32 -> 16


        # bottleneck
        self.bottleneck_conv = nn.Sequential \
            (nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=(1, 1)),
                                             nn.BatchNorm2d(1024),
                                             nn.ReLU(),
                                             nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=(1, 1)),
                                             nn.BatchNorm2d(1024),
                                             nn.ReLU())

        # decoder (upsampling)
        self.upsample0 = nn.ConvTranspose2d(1024, 1024, stride=2, kernel_size=4, padding=1) # 16 -> 32
        self.dec_conv0 = nn.Sequential(nn.Conv2d(in_channels=1536, out_channels=512, kernel_size=3, padding=(1, 1)),
                                       nn.BatchNorm2d(512),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=(1, 1)),
                                       nn.BatchNorm2d(512),
                                       nn.ReLU())
        self.upsample1 = nn.ConvTranspose2d(512, 512, stride=2, kernel_size=4, padding=1) # 32 -> 64
        self.dec_conv1 = nn.Sequential(nn.Conv2d(in_channels=768, out_channels=256, kernel_size=3, padding=(1, 1)),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=(1, 1)),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU())
        self.upsample2 = nn.ConvTranspose2d(256, 256, stride=2, kernel_size=4 ,padding=(1 ,1)) # 64 -> 128
        self.dec_conv2 = nn.Sequential(nn.Conv2d(in_channels=384, out_channels=128, kernel_size=3, padding=(1, 1)),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=(1, 1)),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU())
        self.upsample3 = nn.ConvTranspose2d(128, 128, stride=2, kernel_size=4 ,padding=(1 ,1)) # 128 -> 256
        self.dec_conv3 = nn.Sequential(nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=(1, 1)),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=(1, 1)),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU())
        self.upsample4 = nn.ConvTranspose2d(64, 64, stride=2, kernel_size=4 ,padding=(1 ,1))  # 256 -> 512
        self.dec_conv4 = nn.Sequential(nn.Conv2d(in_channels=96, out_channels=32, kernel_size=3, padding=(1, 1)),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=(1, 1)),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=(1, 1)))

    def forward(self, x):
        # encoder
        e0 = self.enc_conv0(x)
        pool0 = self.pool0(e0)

        e1 = self.enc_conv1(pool0)
        pool1 = self.pool1(e1)

        e2 = self.enc_conv2(pool1)
        pool2 = self.pool2(e2)

        e3 = self.enc_conv3(pool2)
        pool3 = self.pool3(e3)

        e4 = self.enc_conv4(pool3)
        pool4 = self.pool4(e4)

        # bottleneck
        b = self.bottleneck_conv(pool4)

        # decoder
        up0 = self.upsample0(b)
        concat0 = torch.cat([e4, up0] ,dim=1)
        d0 = self.dec_conv0(concat0)

        up1 = self.upsample1(d0)
        concat1 = torch.cat([e3, up1] ,dim=1)
        d1 = self.dec_conv1(concat1)

        up2 = self.upsample2(d1)
        concat2 = torch.cat([e2, up2] ,dim=1)
        d2 = self.dec_conv2(concat2)

        up3 = self.upsample3(d2)
        concat3 = torch.cat([e1, up3] ,dim=1)
        d3 = self.dec_conv3(concat3)

        up4 = self.upsample4(d3)
        concat4 = torch.cat([e0, up4] ,dim=1)
        d4 = self.dec_conv4(concat4)
        return d4