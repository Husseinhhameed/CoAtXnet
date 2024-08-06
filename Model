def conv_3x3_bn(inp, oup, image_size, downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )

class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class XMBConv(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
            self.poolx = nn.MaxPool2d(3, 2, 1)
            self.projx = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
            self.convx = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
            self.convx = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)
        self.convx = PreNorm(inp, self.convx, nn.BatchNorm2d)

    def forward(self, x):
        y = x[1]
        x = x[0]

        if self.downsample:
            out1 = self.proj(self.pool(x)) + self.conv(x)
            out2 = self.projx(self.poolx(y)) + self.convx(y)
        else:
            out1 = x + self.conv(x)
            out2 = y + self.convx(y)
        return [out1, out2]

class XAttention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        self.to_kx = nn.Linear(inp, inner_dim * 1, bias=False)
        self.to_qx = nn.Linear(inp, inner_dim * 1, bias=False)


        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, y):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        kx = self.to_kx(y)
        qx = self.to_qx(x)

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        kx = rearrange(kx, 'b n (h d) -> b h n d', h=self.heads)
        qx = rearrange(qx, 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=self.ih*self.iw, w=self.ih*self.iw)
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)

        dots = torch.matmul(qx, kx.transpose(-1, -2)) * self.scale
        dots = dots + relative_bias
        attn = self.attend(dots)
        out = torch.matmul(attn, out)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class XTransformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0.):
        super().__init__()
        hidden_dim = int(inp * 4)

        self.ih, self.iw = image_size
        self.downsample = downsample

        self.layer_norm = nn.LayerNorm(inp)
        self.layer_normx = nn.LayerNorm(inp)

        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
            self.pool1x = nn.MaxPool2d(3, 2, 1)
            self.pool2x = nn.MaxPool2d(3, 2, 1)
            self.projx = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        self.attn = XAttention(inp, oup, image_size, heads, dim_head, dropout)
        self.ff = FeedForward(oup, hidden_dim, dropout)
        self.attnx = XAttention(inp, oup, image_size, heads, dim_head, dropout)
        self.ffx = FeedForward(oup, hidden_dim, dropout)



        self.ff = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(oup, self.ff, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
            )
        self.ffx = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(oup, self.ffx, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
            )

    def forward(self, x):
        y = x[1]
        x = x[0]

        if self.downsample:
            pool1 = self.pool2(x)
            pool1 = rearrange(pool1, 'b c ih iw -> b (ih iw) c')
            norm1 = self.layer_norm(pool1)
            pool2 = self.pool2x(y)
            pool2 = rearrange(pool2, 'b c ih iw -> b (ih iw) c')
            norm2 = self.layer_normx(pool2)

            attn1 = self.attn(norm1, norm2)
            attn1 = rearrange(attn1, 'b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
            attn2 = self.attnx(norm2, norm1)
            attn2 = rearrange(attn2, 'b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)

            out1 = self.proj(self.pool1(x)) + attn1
            out2 = self.projx(self.pool1x(y)) + attn2
        else:
            xx = rearrange(x, 'b c ih iw -> b (ih iw) c')
            norm1 = self.layer_norm(xx)
            yy = rearrange(y, 'b c ih iw -> b (ih iw) c')
            norm2 = self.layer_normx(yy)

            attn1 = self.attn(norm1, norm2)
            attn1 = rearrange(attn1, 'b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
            attn2 = self.attnx(norm2, norm1)
            attn2 = rearrange(attn2, 'b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
            out1 = x + attn1
            out2 = y + attn2

        out1 = out1 + self.ff(out1)
        out2 = out2 + self.ffx(out2)
        return [out1, out2]


class CoAtXNet(nn.Module):
    def __init__(self, image_size, in_channels, aux_channels, num_blocks, channels, block_types=['C', 'C', 'T', 'T']):
        super().__init__()
        ih, iw = image_size
        block = {'C': XMBConv, 'T': XTransformer}

        self.s0 = self._make_layer(
            conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih // 2, iw // 2))
        self.s0x = self._make_layer(
            conv_3x3_bn, aux_channels, channels[0], num_blocks[0], (ih // 2, iw // 2))

        self.s1 = self._make_layer(
            block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih // 4, iw // 4))
        self.s2 = self._make_layer(
            block[block_types[1]], channels[1], channels[2], num_blocks[2], (ih // 8, iw // 8))
        self.s3 = self._make_layer(
            block[block_types[2]], channels[2], channels[3], num_blocks[3], (ih // 16, iw // 16))
        self.s4 = self._make_layer(
            block[block_types[3]], channels[3], channels[4], num_blocks[4], (ih // 32, iw // 32))

        self.pool = nn.AvgPool2d(ih // 32, 1)
        hidden_dim = 1024
        self.fc1 = nn.Linear(channels[-1] * 2, hidden_dim)
        self.relu = nn.ReLU()  # Activation function between dense layers
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)  # Another dense layer
        self.fc3 = nn.Linear(hidden_dim // 2, 6)  # Final output layer adjusted to target size

    def forward(self, x, y):

        x = self.s0(x)
        y = self.s0x(y)

        xy = self.s1([x,y])
        xy = self.s2(xy)
        xy = self.s3(xy)
        xy = self.s4(xy)

        x = xy[0]
        y = xy[1]

        x = self.pool(x).view(-1, x.shape[1])
        y = self.pool(y).view(-1, y.shape[1])

        x = torch.cat((x,y), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)



def coatxnet_0():
    num_blocks = [2, 2, 3, 5, 2]            # L
    channels = [64, 96, 192, 384, 768]      # D
    return CoAtXNet((256, 256), 3, 1, num_blocks, channels, )


def coatxnet_1():
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [64, 96, 192, 384, 768]      # D
    return CoAtXNet((256, 256), 3, 1, num_blocks, channels, )


def coatxnet_2():
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [128, 128, 256, 512, 1026]   # D
    return CoAtXNet((256, 256), 3, 1, num_blocks, channels, )


def coatxnet_3():
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [192, 192, 384, 768, 1536]   # D
    return CoAtXNet((256, 256), 3, 1, num_blocks, channels, )


def coatxnet_4():
    num_blocks = [2, 2, 12, 28, 2]          # L
    channels = [192, 192, 384, 768, 1536]   # D
    return CoAtXNet((256, 256), 3, 1, num_blocks, channels, )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    img = torch.randn(1, 3, 256, 256)
    aux = torch.randn(1, 1, 256, 256)

    net = coatxnet_0()
    out = net(img, aux)
    print(out.shape, count_parameters(net))

    net = coatxnet_1()
    out = net(img, aux)
    print(out.shape, count_parameters(net))

    net = coatxnet_2()
    out = net(img, aux)
    print(out.shape, count_parameters(net))

    net = coatxnet_3()
    out = net(img, aux)
    print(out.shape, count_parameters(net))

    net = coatxnet_4()
    out = net(img, aux)
    print(out.shape, count_parameters(net))
