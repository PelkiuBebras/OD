from tinygrad import Tensor, nn
from tinygrad.function import Relu
import numpy as np

relu = Relu()


class Identity:
  def __call__(self, x:Tensor) -> Tensor: return x


class RepVGGBlock:
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, deploy=False, use_se=False):
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = Relu()

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = ConvModule(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, activation_type=None, padding=padding, groups=groups)
            self.rbr_1x1 = ConvModule(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, activation_type=None, padding=padding_11, groups=groups)

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _avg_to_3x3_tensor(self, avgp):
        channels = self.in_channels
        groups = self.groups
        kernel_size = avgp.kernel_size
        input_dim = channels // groups
        k = Tensor.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
        return k

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return kernel1x1.pad([1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, ConvModule):
            kernel = branch.conv.weight
            bias = branch.conv.bias
            return kernel, bias
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = Tensor(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True



class ConvModule:
    '''A combination of Conv + BN + Activation'''
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation_type, padding=None, groups=1, bias=False):
        if padding is None: padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.conv_bn = lambda x: nn.BatchNorm2d(out_channels)(self.conv(x))
        self.activation_type = activation_type
    
    def _activate(self, fn: callable, x):
        if self.activation_type is None: return fn(x)
        elif self.activation_type is 'relu': return relu(fn(x))
        elif self.activation_type is 'silu': fn(x).silu()

    def forward(self, x): return self._activate(self.conv_bn, x)
    def forward_fuse(self, x):return self._activate(self.conv, x)


class RepBlock:
    def __init__(self, in_channels, out_channels, n=1, block=RepVGGBlock):
        self.conv1 = block(in_channels, out_channels)
        if n > 1:
            self.block: Tensor = block(out_channels, out_channels)
            for _ in range(n - 2):
                self.block = self.block.sequential(block(out_channels, out_channels))
        else: 
            self.block = None
         
    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x


class ConvBNReLU:
    '''Conv and BN with ReLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        self.block = ConvModule(in_channels, out_channels, kernel_size, stride, 'relu', padding, groups, bias)

    def forward(self, x):
        return self.block(x)


class ConvBNSiLU:
    '''Conv and BN with SiLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        self.block = ConvModule(in_channels, out_channels, kernel_size, stride, 'silu', padding, groups, bias)

    def forward(self, x):
        return self.block(x)

class SPPFModule:
    def __init__(self, in_channels, out_channels, kernel_size=5, block=ConvBNReLU):
        c_ = in_channels // 2  # hidden channels
        self.cv1 = block(in_channels, c_, 1, 1)
        self.cv2 = block(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(x.cat(y1, y2, self.m(y2), dim=1))


class SimSPPF:
    '''Simplified SPPF with ReLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size=5, block=ConvBNReLU):
        self.sppf = SPPFModule(in_channels, out_channels, kernel_size, block)

    def forward(self, x):
        return self.sppf(x)


class SPPF:
    '''SPPF with SiLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size=5, block=ConvBNSiLU):
        self.sppf = SPPFModule(in_channels, out_channels, kernel_size, block)

    def forward(self, x):
        return self.sppf(x)


class CSPSPPFModule:
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, in_channels, out_channels, kernel_size=5, e=0.5, block=ConvBNReLU):
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = block(in_channels, c_, 1, 1)
        self.cv2 = block(in_channels, c_, 1, 1)
        self.cv3 = block(c_, c_, 3, 1)
        self.cv4 = block(c_, c_, 1, 1)

        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.cv5 = block(4 * c_, c_, 1, 1)
        self.cv6 = block(c_, c_, 3, 1)
        self.cv7 = block(2 * c_, out_channels, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y0 = self.cv2(x)
        y1 = self.m(x1)
        y2 = self.m(y1)
        y3 = self.cv6(self.cv5(x1.cat(y1, y2, self.m(y2), dim=1)))
        return self.cv7(y0.cat(y3, dim=1))


class SimCSPSPPF:
    '''CSPSPPF with ReLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size=5, e=0.5, block=ConvBNReLU):
        self.cspsppf = CSPSPPFModule(in_channels, out_channels, kernel_size, e, block)

    def forward(self, x):
        return self.cspsppf(x)


class CSPSPPF:
    '''CSPSPPF with SiLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size=5, e=0.5, block=ConvBNSiLU):
        self.cspsppf = CSPSPPFModule(in_channels, out_channels, kernel_size, e, block)

    def forward(self, x):
        return self.cspsppf(x)


class Transpose(nn.Module):
    '''Normal Transpose, default for upsampling'''
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.upsample_transpose = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True
        )

    def forward(self, x):
        return self.upsample_transpose(x)


class BiFusion(nn.Module):
    '''BiFusion Block in PAN'''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cv1 = ConvBNReLU(in_channels[0], out_channels, 1, 1)
        self.cv2 = ConvBNReLU(in_channels[1], out_channels, 1, 1)
        self.cv3 = ConvBNReLU(out_channels * 3, out_channels, 1, 1)

        self.upsample = Transpose(
            in_channels=out_channels,
            out_channels=out_channels,
        )
        self.downsample = ConvBNReLU(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2
        )

    def forward(self, x):
        x0 = self.upsample(x[0])
        x1 = self.cv1(x[1])
        x2 = self.downsample(self.cv2(x[2]))
        return self.cv3(x0.cat(x1, x2, dim=1))