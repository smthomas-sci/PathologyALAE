from PathALAE.layers import *


# Generator
class ResidualBlockAdaIN(Model):
    """
    Generator block using adaptive instance normalisation to inject style,
    as per StyleGAN version 1
    """
    def __init__(self, filters, block, **kwargs):
        super(ResidualBlockAdaIN, self).__init__(**kwargs)
        # Attributes
        self.filters = filters
        self.block = block
        self.dim = 2 ** (block + 1)

        # Trainable Layers
        # phase 1
        self.beta1 = DenseEQ(units=filters, name=f"G_block_{block}_beta1")
        self.beta1r = Reshape([1, 1, filters], name=f"G_block_{block}_beta1_reshape")
        self.gamma1 = DenseEQ(units=filters, name=f"G_block_{block}_gamma1")
        self.gamma1r = Reshape([1, 1, filters], name=f"G_block_{block}_gamma1_reshape")
        self.noise1 = Conv2DEQ(filters=filters, kernel_size=1, padding='same', name=f"G_block_{block}_noise_bias1")
        self.conv1 = Conv2DEQ(filters=filters, kernel_size=3, padding='same', name=f"G_block_{block}_decoder_conv1")
        self.AdaIn1 = AdaInstanceNormalization(name=f"G_block_{block}_AdaIN_1")
        self.addNoise1 = Add(name=f"G_block_{block}_Add_1")
        self.act1 = LeakyReLU(0.2, name=f"G_block_{block}_Act_1")

        # phase 2
        self.beta2 = DenseEQ(units=filters, name=f"G_block_{block}_beta2")
        self.beta2r = Reshape([1, 1, filters], name=f"G_block_{block}_beta2_reshape")
        self.gamma2 = Dense(units=filters, name=f"G_block_{block}_gamma2")
        self.gamma2r = Reshape([1, 1, filters], name=f"G_block_{block}_gamma2_reshape")
        self.noise2 = Conv2DEQ(filters=filters, kernel_size=1, padding='same', name=f"G_block_{block}_noise_bias2")
        self.conv2 = Conv2DEQ(filters=filters, kernel_size=3, padding='same', name=f"G_block_{block}_decoder_conv2")
        self.AdaIn2 = AdaInstanceNormalization(name=f"G_block_{block}_AdaIN_2")
        self.addNoise2 = Add(name=f"G_block_{block}_Add_2")
        self.act2 = LeakyReLU(0.2, name=f"G_block_{block}_Act_2")

    def call(self, inputs, **kwargs):
        # Unpack inputs
        input_tensor, noise_tensor, style_tensor = inputs

        # Get noise image for level
        noise_tensor = crop_noise(noise_tensor, self.dim, self.block)

        x = input_tensor

        # Phase 1
        beta = self.beta1r(self.beta1(style_tensor))
        gamma = self.gamma1r(self.gamma1(style_tensor))
        noise = self.noise1(noise_tensor)
        x = self.conv1(x)
        x = self.AdaIn1([x, beta, gamma])
        x = self.addNoise1([x, noise])
        x = self.act1(x)

        # Phase 2
        beta = self.beta2r(self.beta2(style_tensor))
        gamma = self.gamma2r(self.gamma2(style_tensor))
        noise = self.noise2(noise_tensor)
        x = self.conv2(x)
        x = self.AdaIn2([x, beta, gamma])
        x = self.addNoise2([x, noise])
        x = self.act2(x)

        return x + input_tensor


class ResidualBlockDeMod(Model):
    """
    Generator block using adaptive instance normalisation to inject style,
    as per StyleGAN version 1
    """
    def __init__(self, filters, block, z_dim, **kwargs):
        super(ResidualBlockDeMod, self).__init__(**kwargs)
        # Attributes
        self.filters = filters
        self.block = block
        self.z_dim = z_dim
        self.dim = 2 ** (block + 1)

        # Trainable Layers
        # phase 1
        self.mod1 = ModulationConv2D(filters=filters,
                                      kernel_size=(3, 3),
                                      style_fmaps=z_dim,
                                      block=block,
                                      demodulate=True,
                                      name=f"G_block_{block}_decoder_demod1")
        self.noise1 = Noise(name=f"G_block_{block}_noise1")
        self.bias1 = Bias(units=filters, name=f"G_block_{block}_decoder_bias1")
        self.act1 = LeakyReLU(0.2, name=f"G_block_{block}_Act_1")

        # phase 2
        self.mod2 = ModulationConv2D(filters=filters,
                                      kernel_size=(3, 3),
                                      style_fmaps=z_dim,
                                      block=block,
                                      demodulate=True,
                                      name=f"G_block_{block}_decoder_demod2")
        self.noise2 = Noise(name=f"G_block_{block}_noise2")
        self.bias2 = Bias(units=filters, name=f"G_block_{block}_decoder_bias2")
        self.act2 = LeakyReLU(0.2, name=f"G_block_{block}_Act_2")

    def call(self, inputs, **kwargs):
        # Unpack inputs
        input_tensor, style_tensor = inputs

        x = input_tensor

        # Phase 1
        x = self.mod1([x, style_tensor])
        x = self.noise1(x)
        x = self.bias1(x)
        x = self.act1(x)

        # Phase 2
        x = self.mod2([x, style_tensor])
        x = self.noise2(x)
        x = self.bias2(x)
        x = self.act2(x)

        # Optional - Make residual
        x += input_tensor

        return x


# Encoder
class ResidualBlockInNorm(Model):
    """
    Encoder block using instance normalisation to extract style
    as introduced in the ALAE by Podgorskiy et al (2020).
    """
    def __init__(self, filters, block, z_dim, **kwargs):
        """
        :param filters: the number of convolution filters (fixed 3x3 size)
        :param block: the block number for naming
        :param z_dim: the z-dimension for mapping features to style vector
        """
        super(ResidualBlockInNorm, self).__init__(**kwargs)
        # Attributes
        self.filters = filters
        self.block = block
        self.z_dim = z_dim

        # Trainable Layers
        self.conv1 = Conv2DEQ(filters=filters, kernel_size=(3, 3), padding="same", name=f"E_block_{block}_Conv_1")
        self.act1 = LeakyReLU(0.2, name=f"E_block_{block}_Act_1")
        self.msd = MeanAndStDev(name=f"E_block_{block}_msd")
        self.in1 = InstanceNormalization(name=f"E_block_{block}_IN_1", center=False, scale=False)
        self.in2 = InstanceNormalization(name=f"E_block_{block}_IN_2", center=False, scale=False)
        self.conv2 = Conv2DEQ(filters=filters, kernel_size=(3, 3), padding="same", name=f"E_block_{block}_Conv_2")
        self.act2 = LeakyReLU(0.2, name=f"E_block_{block}_Act_2")
        self.downsample = AveragePooling2D(name=f"E_block_{block}_DownSample")
        self.mapStyle1 = DenseEQ(units=z_dim, name=f"E_block_{block}_style_1")
        self.mapStyle2 = DenseEQ(units=z_dim, name=f"E_block_{block}_style_2")
        self.flatten = Flatten(name=f"E_block_{block}_flatten")

    def call(self, inputs, **kwargs):
        # Convolution 1
        x = self.conv1(inputs)
        x = self.act1(x)

        # Instance Normalisation 1
        style1 = self.flatten(self.msd(x))
        x = self.in1(x)

        # Convolution 2
        x = self.conv2(x)
        x = self.act2(x)

        # Instance Normalisation 2
        style2 = self.flatten(self.msd(x))
        x = self.in2(x)

        # Affine transform to style vectors
        w1 = self.mapStyle1(style1)
        w2 = self.mapStyle2(style2)

        # Make residual
        x += inputs

        return x, w1, w2


# Encoder / Decoder Block (non-style)
class ResidualBlock(Model):
    """
    Encoder block using instance normalisation to extract style
    as introduced in the ALAE by Podgorskiy et al (2020).
    """
    def __init__(self, filters, block, **kwargs):
        """
        :param filters: the number of convolution filters (fixed 3x3 size)
        :param block: the block number for naming
        """
        super(ResidualBlock, self).__init__(**kwargs)
        # Attributes
        self.filters = filters
        self.block = block

        # Trainable Layers
        self.conv1 = Conv2DEQ(filters=filters, kernel_size=(3, 3), padding="same", name=f"E_block_{block}_Conv_1")
        self.bn1 = BatchNormalization(name=f"E_block_{block}_BN_1")
        self.act1 = LeakyReLU(0.2, name=f"E_block_{block}_Act_1")

        self.conv2 = Conv2DEQ(filters=filters, kernel_size=(3, 3), padding="same", name=f"E_block_{block}_Conv_2")
        self.bn2 = BatchNormalization(name=f"E_block_{block}_BN_2")
        self.act2 = LeakyReLU(0.2, name=f"E_block_{block}_Act_2")

    def call(self, inputs, **kwargs):
        # Convolution 1
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)

        # Convolution 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        # Make residual
        x += inputs

        return x
