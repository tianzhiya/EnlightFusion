import torch
import torch.nn as nn

input_tensor = torch.randn(1, 1, 256, 256)


class UpsampleModel(nn.Module):
    def __init__(self, height, width):
        super(UpsampleModel, self).__init__()
        self.conv = nn.Conv2d(1, 48, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(size=(height, width), mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x


model = UpsampleModel(640, 480)

output_tensor = model(input_tensor)

print(output_tensor.size())
