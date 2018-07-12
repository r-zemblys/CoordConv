import torch
import torch.nn as nn


class AddCoords(nn.Module):
    def __init__(self, x_dim=64, y_dim=64, with_r=False):
        super(AddCoords, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        input_tensor: (batch, c, x_dim, y_dim)
        """
        batch_size_tensor = input_tensor.shape[0]
        
        xx_ones = torch.ones([self.x_dim, 1], dtype=torch.int32)
        xx_range = torch.arange(self.y_dim, dtype=torch.int32).unsqueeze(0)
        xx_channel = torch.matmul(xx_ones, xx_range).unsqueeze(0)
        
        yy_ones = torch.ones([1, self.y_dim], dtype=torch.int32)
        yy_range = torch.arange(self.x_dim, dtype=torch.int32).unsqueeze(1)
        yy_channel = torch.matmul(yy_range, yy_ones).unsqueeze(0)
        
        xx_channel = xx_channel.float() / (self.y_dim - 1)
        yy_channel = yy_channel.float() / (self.x_dim - 1)
        
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1
        

        xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1)
        
        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):
    """CoordConv layer as in the paper."""
    def __init__(self, x_dim, y_dim, with_r, *args, **kwargs):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(x_dim=x_dim, y_dim=y_dim, with_r=with_r)
        self.conv = nn.Conv2d(*args, **kwargs)

    def forward(self, input_tensor):
        ret = self.addcoords(input_tensor)
        ret = self.conv(ret)
        return ret