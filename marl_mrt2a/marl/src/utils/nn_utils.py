import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod
try:
    from . import resnet
except:
    import resnet
try:
    from . import unet
except:
    import unet
    

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode="bilinear"):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        # self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x


net_config = {
    "resnet": {
        "class": resnet.resnet18,
        "kwargs": ["num_input_channels"],
        "inferFirst": True,
        },
    "unet": {
        "class": unet.UNet, 
        "kwargs": ["shape", "in_channels", "out_channels", "min_size"],
        "inferFirst": True,
        },
    "conv2d": {
        "class": nn.Conv2d,
        "kwargs": ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
        "inferFirst": True,
        },
    "batchNorm2d": {
        "class": nn.BatchNorm2d, 
        "kwargs": ["num_features"],
        "inferFirst": True,
        },
    "relu": {
        "class": nn.ReLU, 
        "kwargs": [],
        },
    "avgPool2d": {
        "class": nn.AvgPool2d, 
        "kwargs": ["kernel_size", "stride", "padding"],
        },
    "linear": {
        "class": nn.Linear, 
        "kwargs": ["in_features", "out_features","bias"],
        "inferFirst": True,
        },
    "batchNorm1d": {
        "class": nn.BatchNorm1d, 
        "kwargs": ["num_features"],
        "inferFirst": True,
        },
    "flatten": {
        "class": lambda: nn.Flatten(start_dim=-3), 
        "kwargs": [],
        },
    "interpolate": {
        "class": Interpolate, 
        "kwargs": ["scale_factor"],
        },
}

def layer_from_string(layer_str, input_shape):
    args = layer_str.split(",")
    layer_type = args[0]
    assert layer_type in net_config.keys(), f"Unexpected layer type: {layer_type}"

    kwargs = {}
    if net_config[layer_type].get("inferFirst", False):
        if net_config[layer_type]["kwargs"][0]=="shape":
            values = [str(input_shape)]
        else:
            values = [str(input_shape[0])]
    else:
        values = []
    if len(args)>1:
        values = values + args[1:]
    for i in range(len(values)):
        kwargs[net_config[layer_type]["kwargs"][i]] = eval(values[i])

    layer = net_config[layer_type]["class"](**kwargs)
    
    x = torch.rand(input_shape)
    if layer_type.startswith("batchNorm"):
        output_shape = input_shape
    elif layer_type=="flatten":
        output_shape = (prod(input_shape),)
    elif layer_type in ["resnet","unet","interpolate","conv2d"]:
        x = x.unsqueeze(0)
        output_shape = layer(x).shape[1:]
    else:
        output_shape = layer(x).shape

    return layer, output_shape

def net_from_string(string, input_shape, target_shape=None):
    # Given a string, return the corresponding torch.nn.Sequential
    # Separate between CNN and MLP parts
    strs = string.split("&")
    assert len(strs)==2, f"Expected CNN and MLP strings, separated by '&', but got: {string}"
    cnn_str, mlp_str = strs

    if target_shape=="same":
        # Output shape should be the same as the input, but with 1 channel only
        # Ensure there are no linear layers
        assert len(mlp_str)==0, f"Specified target_shape as grid, but architecture has linear layers. str='{string}'"
        raise NotImplementedError("This option does not support comms")
        target_shape = (1, *input_shape[1:])

    # Create CNN
    cnn_str = cnn_str.split(";")
    if cnn_str == [""]: cnn_str=[]
    layers = []
    for l in cnn_str:
        layer, input_shape = layer_from_string(l, input_shape=input_shape)
        layers.append(layer)

    mlp_str = mlp_str.split(";")
    if mlp_str==[""]: mlp_str=[]

    if target_shape is not None:
        if len(target_shape)>1:
            # assert output_shape==target_shape, "Output shape must be equal to input shape"
            if input_shape[1:]==target_shape[1:]:
                l = f"conv2d,{target_shape[0]},1,1,0"
                layer, input_shape = layer_from_string(l, input_shape=input_shape)
                layers.append(layer)
            else:
                raise ValueError("Output shape must be equal to input shape (not counting number of channels)")
                # Add interpolation layer, if necessary
                # TODO
        else:
            assert len(target_shape)==1, "Target shape should be the output of a linear layer"
            mlp_str += [f"linear,{target_shape[0]},0"]
    
    if len(mlp_str)==0:
        return nn.Sequential(*layers), input_shape
    # layer, output_shape = layer_from_string("flatten", input_shape=input_shape)
    # layers.append(layer)
    # input_shape = output_shape
    mlp_str = ["flatten"]+mlp_str

    for l in mlp_str:
        layer, input_shape = layer_from_string(l, input_shape=input_shape)
        layers.append(layer)

    # if target_shape is not None:
    #     assert len(target_shape)==1, "Target shape should be the output of a linear layer"
    #     layer, output_shape = layer_from_string(f"linear,{target_shape[0]},0", input_shape=input_shape)
    #     layers.append(layer)
    
    return nn.Sequential(*layers), input_shape

def check_valid_string(string):

    strs = string.split(" & ")
    assert len(strs)==2, f"Expected CNN and MLP strings, separated by '&', but got: {string}"
    cnn_str, mlp_str = strs
    cnn_str = cnn_str.split(";")



if __name__=="__main__":
    # default_str="batchNorm2d;conv2d,64,5,1,2;relu;conv2d,64,5,1,2;batchNorm2d;conv2d,32,5,1,2;relu;conv2d,32,5,1,2;batchNorm2d;conv2d,16,3,1,1;relu;conv2d,16,3,1,1;relu;conv2d,16,3,1,1;relu;conv2d,1,3,1,1&"
    # default_str="batchNorm2d;conv2d,64,5,1,2;relu;conv2d,64,5,1,2;batchNorm2d;conv2d,32,3,1,1;relu;conv2d,32,3,1,1;relu;conv2d,16,3,1,1;relu;conv2d,1,3,1,1&"
    # default_str="batchNorm2d;conv2d,32,5,1,2;relu;conv2d,64,5,1,2;batchNorm2d;conv2d,16,3,1,1;relu;conv2d,1,3,1,1&"
    default_str = "unet,8,1,2&"
    input_shape = (8,3,3)

    # net, out = net_from_string(default_str, input_shape=input_shape)
    net, out = net_from_string(default_str, input_shape=input_shape, target_shape="same")
    # res = net_from_string(default_str, input_shape=(3,9,9))

    v = torch.rand(10,*input_shape)
    y = net(v)
    
    print(net)