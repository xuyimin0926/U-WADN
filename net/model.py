from torch import nn

from net.encoder import CBDE
from net.DGRN import DGRN
from net.slimmable_op import SlimmableConv2d


class AirNet(nn.Module):
    def __init__(self, opt, width_mult_list=[0.25, 0.5, 0.75, 1.0]):
        super(AirNet, self).__init__()

        # Restorer
        self.R = DGRN(opt, width_mult_list=width_mult_list)

        # Encoder
        self.E = CBDE(opt)

        # Transforms
        self.transform = SlimmableConv2d(
            in_channels_list=[64 for _ in width_mult_list],
            out_channels_list=[int(64 * width_mult) for width_mult in width_mult_list],
            width_mult_list=width_mult_list,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False
        )
        
        self.stage_2 = opt.stage == 2
        self.width_mult_list = width_mult_list
        self.curr_ratio = 1.0

    def forward(self, x_query, x_key, forward_E=False):
        if self.training:
            if forward_E:
                fea, logits, inter = self.E(x_query) 
                return fea, logits, inter
            else:
                if not self.stage_2:
                    fea, logits, inter = self.E(x_query)
                    inter = self.transform(inter)
                    restored = self.R(x_query, inter)
    
                    return restored, logits
                else:
                    fea, logits, inter, selection = self.E(x_query)
                    inter = self.transform(inter)
                    restored = self.R(x_query, inter)
    
                    return restored, logits, selection
        else:
            if self.stage_2:
                fea, inter, selection = self.E(x_query)
                self.set_slimmable_ratio(self.width_mult_list[selection])
                inter = self.transform(inter)
                restored = self.R(x_query, inter)

                return restored, self.width_mult_list[selection]
            else:
                fea, inter, selection = self.E(x_query)
                inter = self.transform(inter)
                restored = self.R(x_query, inter)
                return restored

    def set_slimmable_ratio(self, slimmable_ratio):
        self.curr_ratio = slimmable_ratio
        for name, module in self.named_modules():
            if hasattr(module, 'width_mult'):
                setattr(module, 'width_mult', slimmable_ratio)
    
    def fix_gradient(self):
        self.E.fix_gradient()
        self.R.requires_grad_(False)
