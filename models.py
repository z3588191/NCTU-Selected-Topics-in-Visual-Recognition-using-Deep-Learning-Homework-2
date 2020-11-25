import torch
from collections import OrderedDict
import torch.nn as nn
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool, FeaturePyramidNetwork
import timm


def _conv2d(in_channels,out_channels,size = 1):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels,out_channels,size,stride=1,padding=size//2),
        torch.nn.BatchNorm2d(out_channels)
        )


class ExactFusionModel(torch.nn.Module):
    def __init__(self, in_channels_list, out_channels, transition=128, withproduction = True, extra_blocks=None):
        if len(in_channels_list) < 4:
            raise('lenght of in_channels_list must be longer than 3')
        super(ExactFusionModel, self).__init__()
        self.in_channels_list = in_channels_list
        self.same_blocks = nn.ModuleList()
        self.prod_blocks = nn.ModuleList()
        self.upto_blocks = nn.ModuleList()
        self.extra_blocks = extra_blocks

        b_index = len(in_channels_list) - 1
        up_channel = self.in_channels_list[b_index] + self.in_channels_list[b_index - 1] // 2
        self.efm_channels = [up_channel]

        b_index -= 1
        while b_index > 0:
            channels = self.in_channels_list[b_index]//2 + self.in_channels_list[b_index - 1] // 2 + transition
            up_channel = channels
            self.efm_channels.insert(0, channels)
            b_index -= 1

        for i, in_channel in enumerate(self.efm_channels):
            self.same_blocks.append(_conv2d(in_channel,out_channels,3))
            self.prod_blocks.append(_conv2d(out_channels,out_channels,3) if withproduction else torch.nn.Identity())
            self.upto_blocks.append(_conv2d(in_channel,transition,1))

        for m in self.children():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight, a=1)
                torch.nn.init.constant_(m.bias, 0)

    #input must be dict,from bottom to top
    def forward(self,x):
        names = list(x.keys())
        x = list(x.values())

        xb_index = len(x) - 1
        shape = x[xb_index].shape[-2:]
        csp_x = [torch.cat(
            [torch.nn.functional.interpolate(x[xb_index - 1][:, self.in_channels_list[xb_index - 1]//2:]
                                                            , size=shape, mode="nearest"), x[xb_index]], 1)]
        xb_index -= 1

        while xb_index > 0:
            shape = x[xb_index].shape[-2:]
            csp_x.insert(0,
                        torch.cat([torch.nn.functional.interpolate(
                        x[xb_index - 1][:, self.in_channels_list[xb_index - 1]//2:], size=shape, mode='nearest'),
                        x[xb_index][:, :self.in_channels_list[xb_index]//2],
                        torch.nn.functional.interpolate(self.upto_blocks[xb_index](csp_x[0]), size=shape, mode='nearest')], 1))
            xb_index -= 1

        bottom_feature = self.same_blocks[0](csp_x[0])
        result = [self.prod_blocks[0](bottom_feature)]
        for csp, same_block, prod_block in zip(csp_x[1:],self.same_blocks[1:],self.prod_blocks[1:]):
            shape = csp.shape[-2:]
            feature = same_block(csp)
            feature = feature + torch.nn.functional.interpolate(bottom_feature, size=shape, mode='nearest')
            bottom_feature = feature
            result.append(prod_block(feature))

        if self.extra_blocks is not None:
            result, names = self.extra_blocks(result, x, names)

        out = OrderedDict((k,v) for k ,v in zip(names[1:],result))

        return out


class CSPIntermediateLayer(nn.ModuleDict):
    def __init__(self, model: nn.Module) -> None:

        layers = OrderedDict()
        layers['stem'] = model.stem
        layers['0'] = model.stages[0]
        layers['1'] = model.stages[1]
        layers['2'] = model.stages[2]
        layers['3'] = model.stages[3]

        super(CSPIntermediateLayer, self).__init__(layers)

        self.return_layers = ['0','1','2','3']

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out[name] = x
        return out


class CSPWithFPN(nn.Module):
    def __init__(self):
        super(CSPWithFPN, self).__init__()

        extra_blocks = LastLevelMaxPool()

        backbone = timm.models.cspresnet50(pretrained=True)

        layers_to_train = ['stages.1', 'stages.2', 'stages.3']
        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)

        self.body = CSPIntermediateLayer(backbone)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[128, 256, 512, 1024],
            out_channels=256,
            extra_blocks=extra_blocks,
        )
        self.out_channels = 256

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x


class CSPWithEFM(nn.Module):
    def __init__(self):
        super(CSPWithEFM, self).__init__()

        extra_blocks = LastLevelMaxPool()

        backbone = timm.models.cspresnet50(pretrained=True)

        layers_to_train = ['stages.1', 'stages.2', 'stages.3']
        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)

        self.body = CSPIntermediateLayer(backbone)
        self.fpn = ExactFusionModel(
            in_channels_list=[128, 256, 512, 1024],
            out_channels=256,
            extra_blocks=extra_blocks,
        )
        self.out_channels = 256

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x
