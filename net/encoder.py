import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_feat, out_feat, stride=1):
        super(ResBlock, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_feat),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_feat),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_feat)
        )

    def forward(self, x):
        return nn.LeakyReLU(0.1, True)(self.backbone(x) + self.shortcut(x))


class ResEncoder(nn.Module):
    def __init__(self):
        super(ResEncoder, self).__init__()

        self.E_pre = ResBlock(in_feat=3, out_feat=64, stride=1)
        self.E = nn.Sequential(
            ResBlock(in_feat=64, out_feat=128, stride=2),
            nn.AdaptiveAvgPool2d(1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1, True),
            nn.Linear(128, 5),
        )

    def forward(self, x):
        inter = self.E_pre(x)
        fea = self.E(inter).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)

        return fea, out, inter


class CBDE(nn.Module):
    def __init__(self, opt):
        super(CBDE, self).__init__()

        dim = 256

        # Encoder
        self.E = ResEncoder()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 16, 1, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(16, 64, 1, bias=False),
        )

        self.is_stage_2 = opt.stage == 2
        self.classifier = nn.Linear(64, 5)

    def forward(self, x_query):
        if self.training:
            # degradation-aware represenetion learning

            if self.is_stage_2:
                fea, logits, inter = self.E(x_query)
                final_logits = logits + 0.5 * self.classifier(self.se(inter).squeeze())
                return fea, logits, inter, final_logits
            else:
                fea, logits, inter = self.E(x_query)
                return fea, logits, inter
        else:
            # degradation-aware represenetion learning
            fea, logits, inter = self.E(x_query)
            final_logits = logits + 0.5 * self.classifier(self.se(inter).squeeze())
            selection = final_logits.argmax(dim=1)
            return fea, inter, selection
