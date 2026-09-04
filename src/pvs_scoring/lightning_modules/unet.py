from monai.networks.nets.dynunet import DynUNet
import torch.nn.functional as F

class MyDynUNet(DynUNet):
    
    def __init__(self, deep_supervision_upsample: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.deep_supervision_upsample = deep_supervision_upsample

    def forward(self, x):
        '''
        Not interpolating intermediate feature maps to the output size. The return type during training, when
        deep supervision is on is now a tuple instead of a tensor. This means the model won't  work with torchscript.
        If torchscript is required, suggest padding with -1 and then manually cropping away afterwards.
        '''
        out = self.skip_layers(x)
        out = self.output_block(out)
        if self.training and self.deep_supervision:
            out_all = [out]
            for feature_map in self.heads:
                if self.deep_supervision_upsample:
                    out_all.append(F.interpolate(feature_map, out.shape[2:]))
                else:   
                    out_all.append(feature_map)
            return tuple(out_all)
        return out