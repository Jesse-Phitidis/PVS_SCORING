from monai.networks.nets import ResNet
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.networks.nets.resnet import ResNetBlock, ResNetBottleneck, get_avgpool
from monai.utils import ensure_tuple_rep
from monai.networks.layers.factories import Conv, Pool
import torch.nn as nn
import torch
import inspect
from typing import Callable


def get_all_args(func, args, kwargs):
    sig = inspect.signature(func)
    bound_args = sig.bind_partial(*args, **kwargs)
    bound_args.apply_defaults()
    return bound_args.arguments


class BaseResNet(ResNet):
    
    '''
    Fixing an issue whereby the norm option is no passed into the make layers in MONAI code.
    '''

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        args = get_all_args(super().__init__, args, kwargs)

        block = args["block"]
        if isinstance(block, str):
            if block == "basic":
                block = ResNetBlock
            elif block == "bottleneck":
                block = ResNetBottleneck
            else:
                raise ValueError("Unknown block '%s', use basic or bottleneck" % block)
            
        block_inplanes = args["block_inplanes"]
        layers = args["layers"]
        spatial_dims = args["spatial_dims"]
        shortcut_type = args["shortcut_type"]
        norm = args["norm"]
        self.in_planes = block_inplanes[0]
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], spatial_dims, shortcut_type, norm=norm)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], spatial_dims, shortcut_type, stride=2, norm=norm)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], spatial_dims, shortcut_type, stride=2, norm=norm)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], spatial_dims, shortcut_type, stride=2, norm=norm)
       
       
class MultiKernelSumConv(nn.Module):
    
    def __init__(self, spatial_dims: int, conv_type: nn.Module, n_input_channels: int, in_planes: int, conv1_t_size: list, conv1_t_stride: int) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        for k in conv1_t_size:
            k, conv1_t_stride = ensure_tuple_rep(k, spatial_dims), ensure_tuple_rep(conv1_t_stride, spatial_dims)
            conv = conv_type(
                n_input_channels, 
                in_planes, 
                kernel_size=k, 
                stride=conv1_t_stride, 
                padding=tuple(kx // 2 for kx in k), 
                bias=False
            )
            self.convs.append(conv)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for conv in self.convs:
            outs.append(conv(x))
        return torch.sum(torch.stack(outs, dim=0), dim=0) 
  
        
class MyResNet(BaseResNet):
    
    '''
    Adding option to have the multikernel sum as the first conv.
    '''
    
    def __init__(self, conv1_t_sizes: list[int], *args, **kwargs) -> None:
        print("Using MultiKernelSumConv for conv1. conv1_t_size will be ignored. Use conv1_t_sizes to specify the kernel sizes.")
        super().__init__(*args, **kwargs)
        args = get_all_args(ResNet.__init__, args, kwargs)
        spatial_dims = args["spatial_dims"]
        n_input_channels = args["n_input_channels"]
        in_planes = args["block_inplanes"][0]
        conv1_t_stride = args["conv1_t_stride"]
        if conv1_t_sizes is None:
            conv1_t_sizes = [args["conv1_t_size"]]
        conv_type: type[nn.Conv1d | nn.Conv2d | nn.Conv3d] = Conv[Conv.CONV, spatial_dims]
        self.conv1 = MultiKernelSumConv(
            spatial_dims,
            conv_type,
            n_input_channels,
            in_planes,
            conv1_t_sizes,
            conv1_t_stride
        )
        

class ResNetEncoder(MyResNet):
    
    '''
    Removing the final classification bits and returning features from each of the 4 layers.
    '''

    def __init__(self, conv1_t_sizes: list[int], *args, **kwargs) -> None:
        super().__init__(conv1_t_sizes, *args, **kwargs)
        del self.avgpool
        del self.fc

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        outs = []
        x = self.layer1(x)
        outs.append(x)
        x = self.layer2(x)
        outs.append(x)
        x = self.layer3(x)
        outs.append(x)
        x = self.layer4(x)
        outs.append(x)

        return outs
    
    
class ResNetClassifier(nn.Module):
    
    '''
    Adding option for multiple classification heads
    '''
    
    def __init__(self, fc_num_classes: list[int], conv1_t_sizes: list[int] = [7], *args, **kwargs) -> None:
        super().__init__()
        all_args = get_all_args(ResNet.__init__, args, kwargs)
        
        spatial_dims = all_args["spatial_dims"]
        block = all_args["block"]
        if isinstance(block, str):
            if block == "basic":
                block = ResNetBlock
            elif block == "bottleneck":
                block = ResNetBottleneck
            else:
                raise ValueError("Unknown block '%s', use basic or bottleneck" % block)
        block_inplanes = all_args["block_inplanes"]
        avgp_type: type[nn.AdaptiveAvgPool1d | nn.AdaptiveAvgPool2d | nn.AdaptiveAvgPool3d] = Pool[
            Pool.ADAPTIVEAVG, spatial_dims
        ]
        block_avgpool = get_avgpool()
        
        self.encoder = ResNetEncoder(conv1_t_sizes, *args, **kwargs)
        self.avgpool = avgp_type(block_avgpool[spatial_dims])
        self.fcs = nn.ModuleList()
        for n_classes in fc_num_classes:
            self.fcs.append(nn.Linear(block_inplanes[3] * block.expansion, n_classes))
            
    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        feats = self.encoder(x)
        
        x = self.avgpool(feats[-1])
        x = x.view(x.size(0), -1)
        
        preds = []
        for fc in self.fcs:
            preds.append(fc(x))
            
        return feats, preds
    
    
class ResNetBlockUp(nn.Module):
    
    def __init__(self, in_planes_below, in_planes_skip, **kwargs) -> None:
        super().__init__()
        all_args = get_all_args(ResNetBlock.__init__, [], kwargs)
        
        spatial_dims = all_args["spatial_dims"]
        norm = all_args["norm"]
        act = all_args["act"]
        channels = all_args["planes"]
        conv_type_upsample: Callable = Conv[Conv.CONVTRANS, spatial_dims]
        conv_type_reduce_channels: Callable = Conv[Conv.CONV, spatial_dims]
        
        self.upsample = nn.Sequential(
            conv_type_upsample(in_planes_below, in_planes_skip, kernel_size=2, stride=2, bias=False),
            get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_planes_skip),
            get_act_layer(name=act)
        )
        
        
        reduce_channels = nn.Sequential(
            conv_type_reduce_channels(in_planes_skip * 2, channels, kernel_size=1, stride=1),
            get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=channels)
        )
        
        most_args = {k: v for k, v in kwargs.items() if k not in ["stride", "downsample"]} # Not applicable for this block
        self.resblock = ResNetBlock(in_planes=in_planes_skip * 2, downsample=reduce_channels, **most_args)
        
    def forward(self, x_below: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x_up = self.upsample(x_below)
        x_out = self.resblock(torch.cat([x_up, x_skip],dim=1))
        return x_out
    
    
class ResNetBottleneckUp(nn.Module):
    
    def __init__(self, in_planes_below, in_planes_skip, **kwargs) -> None:
        super().__init__()
        all_args = get_all_args(ResNetBottleneck.__init__, [], kwargs)
        
        spatial_dims = all_args["spatial_dims"]
        norm = all_args["norm"]
        act = all_args["act"]
        channels = all_args["planes"]
        conv_type_upsample: Callable = Conv[Conv.CONVTRANS, spatial_dims]
        conv_type_reduce_channels: Callable = Conv[Conv.CONV, spatial_dims]
        
        self.upsample = nn.Sequential(
            conv_type_upsample(in_planes_below, in_planes_skip, kernel_size=2, stride=2, bias=False),
            get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_planes_skip),
            get_act_layer(name=act)
        )
        
        reduce_channels = nn.Sequential(
            conv_type_reduce_channels(in_planes_skip * 2, channels, kernel_size=1, stride=1),
            get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=channels)
        )
        
        most_args = {k: v for k, v in kwargs.items() if k not in ["stride", "downsample"]} # Not applicable for this block
        self.resblock = ResNetBottleneck(in_panes=in_planes_skip * 2, downsample=reduce_channels, **most_args)
        
    def forward(self, x_below: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x_up = self.upsample(x_below)
        x_out = self.resblock(torch.cat([x_up, x_skip],dim=1))
        return x_out
    

class LightweightResNetDecoder(nn.Module):
    
    def __init__(self, out_channels: int, deep_supervision: bool = False, *args, **kwargs) -> None:
        super().__init__()
        all_args = get_all_args(ResNet.__init__, args, kwargs)
        
        block = all_args["block"]
        if isinstance(block, str):
            if block == "basic":
                block = ResNetBlockUp
            elif block == "bottleneck":
                block = ResNetBottleneckUp
            else:
                raise ValueError("Unknown block '%s', use basic or bottleneck" % block)
        
        self.out_channels = out_channels
        self.deep_supervision = deep_supervision
        
        out_channels = all_args["block_inplanes"][::-1]
        spatial_dims = all_args["spatial_dims"]
        act = all_args["act"]
        norm = all_args["norm"]
        
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        self.layers = nn.ModuleList()
        self.output_layers = nn.ModuleList()
        
        for i in range(3):
            
            layer = block(
                in_planes_below=out_channels[i],
                in_planes_skip=out_channels[i+1],
                planes=out_channels[i+1],
                spatial_dims=spatial_dims,
                act=act,
                norm=norm
            )
            
            self.layers.append(layer)
            
            output_layer = conv_type(out_channels[i+1], self.out_channels, kernel_size=1, stride=1) if (i==2 or deep_supervision) else None
            self.output_layers.append(output_layer)
        
        
    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor | None]:
        outs = []
        for i, (layer, output_layer) in enumerate(zip(self.layers, self.output_layers)):
            feats[i+1] = layer(feats[i], feats[i+1])
            outs.append(output_layer(feats[i+1]) if output_layer else None)
        return outs
        

class ResNetWithSegHead(nn.Module):
    
    '''
    Adding decoder with skips for using segmentation. Note this is for the purpose of using 
    the segmentation oututs as an auxiliary task. To do segmentation only, use DynUNet/SegResNet.
    '''
    
    def __init__(self, out_channels: int, fc_num_classes: list[int], conv1_t_sizes: list[int] = [7], deep_supervision: bool = False, *args, **kwargs) -> None:
        super().__init__()
        self.classifier = ResNetClassifier(fc_num_classes, conv1_t_sizes, *args, **kwargs)
        self.segmentor = LightweightResNetDecoder(out_channels, deep_supervision, *args, **kwargs)
        
    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor | None]]:
        feats, preds = self.classifier(x)
        segs = self.segmentor(feats[::-1])
        return feats, preds, segs