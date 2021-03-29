import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import Dense, LeakyReLU, Conv2D, Flatten, Add, Cropping2D, Layer, InputSpec
from tensorflow.keras.layers import UpSampling2D, Reshape, AveragePooling2D, Input, BatchNormalization
from tensorflow.keras.models import Model


# Initializers
orthogonal = tf.initializers.Orthogonal
normal = tf.initializers.RandomNormal
ones = tf.initializers.ones


def crop_noise(noise_tensor, size, block):
    """
    Crops the noise_tensor to the target size.
    """
    cut = (noise_tensor.shape[1] - size) // 2
    crop = Cropping2D(cut, name=f"G_Noise_Crop_block_{block}")(noise_tensor)
    return crop


class DenseEQ(Dense):
    """
    Standard dense layer but includes learning rate equilization
    at runtime as per Karras et al. 2017. Includes learning rate multiplier,
    but defaults to 1.0. Only needed for the mapping network.

    Inherits Dense layer and overides the call method.
    """
    def __init__(self, lrmul=1, **kwargs):
        if 'kernel_initializer' in kwargs:
            raise Exception("Cannot override kernel_initializer")
        self.lrmul=lrmul
        super().__init__(kernel_initializer=normal(0, 1/self.lrmul), **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        # The number of inputs
        n = np.product([int(val) for val in input_shape[1:]])
        # He initialisation constant
        self.c = np.sqrt(2/n)*self.lrmul

    def call(self, inputs):
        output = K.dot(inputs, self.kernel*self.c)  # scale kernel
        if self.use_bias:
            output = K.bias_add(output, self.bias * self.lrmul, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output


class Conv2DEQ(Conv2D):
    """
    Standard Conv2D layer but includes learning rate equilization
    at runtime as per Karras et al. 2017.

    Inherits Conv2D layer and overrides the call method, following
    https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py

    from the tf-keras branch.

    """

    def __init__(self, lrmul=1, **kwargs):
        """
        Requires usual Conv2D inputs e.g.
         - filters, kernel_size, strides, padding
        """
        self.lrmul = lrmul

        if 'kernel_initializer' in kwargs:
            raise Exception("Cannot override kernel_initializer")
        super().__init__(kernel_initializer=normal(0, 1/self.lrmul), **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        # The number of inputs
        n = np.product([int(val) for val in input_shape[1:]])
        # He initialisation constant
        self.c = np.sqrt(2 / n)*self.lrmul

    def call(self, inputs):
        outputs = K.conv2d(
            inputs,
            self.kernel * self.c,  # scale kernel
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias * self.lrmul,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class AdaInstanceNormalization(Layer):
    """
    This is the AdaInstanceNormalization layer used by manicman199 available at
     https://github.com/manicman1999/StyleGAN-Keras

     This is used in StyleGAN version 1 as well as ALAE.
    """

    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 **kwargs):
        super(AdaInstanceNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center # always done
        self.scale = scale # always done

    def build(self, input_shape):

        dim = input_shape[0][self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                                                        'input tensor should have a defined dimension '
                                                        'but the layer received an input with shape ' +
                             str(input_shape[0]) + '.')

        super(AdaInstanceNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs[0])
        reduction_axes = list(range(0, len(input_shape)))

        beta = inputs[1]
        gamma = inputs[2]

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]
        mean = K.mean(inputs[0], reduction_axes, keepdims=True)
        stddev = K.std(inputs[0], reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs[0] - mean) / stddev

        return normed * gamma + beta

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale
        }
        base_config = super(AdaInstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class Attention(Layer):
    def __init__(self, ch, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.channels = ch
        self.filters_f_g = self.channels // 8
        self.filters_h = self.channels

    def build(self, input_shape):
        kernel_shape_f_g = (1, 1) + (self.channels, self.filters_f_g)
        kernel_shape_h = (1, 1) + (self.channels, self.filters_h)

        # Create a trainable weight variable for this layer:
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.kernel_f = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_f')
        self.kernel_g = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_g')
        self.kernel_h = self.add_weight(shape=kernel_shape_h,
                                        initializer='glorot_uniform',
                                        name='kernel_h')
        self.bias_f = self.add_weight(shape=(self.filters_f_g,),
                                      initializer='zeros',
                                      name='bias_F')
        self.bias_g = self.add_weight(shape=(self.filters_f_g,),
                                      initializer='zeros',
                                      name='bias_g')
        self.bias_h = self.add_weight(shape=(self.filters_h,),
                                      initializer='zeros',
                                      name='bias_h')
        super(Attention, self).build(input_shape)
        # Set input spec.
        self.input_spec = InputSpec(ndim=4,
                                    axes={3: input_shape[-1]})
        self.built = True

    def call(self, x, **kwargs):
        def hw_flatten(x):
            return K.reshape(x, shape=[K.shape(x)[0], K.shape(x)[1] * K.shape(x)[2], K.shape(x)[-1]])

        f = K.conv2d(x,
                     kernel=self.kernel_f,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        f = K.bias_add(f, self.bias_f)
        g = K.conv2d(x,
                     kernel=self.kernel_g,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        g = K.bias_add(g, self.bias_g)
        h = K.conv2d(x,
                     kernel=self.kernel_h,
                     strides=(1, 1), padding='same')  # [bs, h, w, c]
        h = K.bias_add(h, self.bias_h)

        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = K.softmax(s, axis=-1)  # attention map

        self.beta = beta

        o = K.batch_dot(beta, hw_flatten(h))  # [bs, N, C]

        o = K.reshape(o, shape=K.shape(x))  # [bs, h, w, C]
        x = self.gamma * o + x

        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class tRGB(Layer):
    """
    Linear transformation from feature space to rgb space, using a 1x1 convolution
    """
    def __init__(self, block):
        super(tRGB, self).__init__()
        self.block = block
        self.transform = Conv2DEQ(filters=3, kernel_size=(1, 1), padding="same", name=f"G_block_{block}_tRGB")

    def build(self, input_shape):
        super(tRGB, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return self.transform(inputs)


class MeanAndStDev(Layer):
    """
    This is the Instance Normalization transformation which
    concatenates mu and sigma to later be mapped to w.

    This is used in the encoder introduced by
    """
    def __init__(self, **kwargs):
        super(MeanAndStDev, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MeanAndStDev, self).build(input_shape)

    def call(self, inputs, **kwargs):
        m = K.mean(inputs, axis=[1, 2], keepdims=True)
        std = K.std(inputs, axis=[1, 2], keepdims=True)
        statistics = K.concatenate([m, std], axis=1)
        return statistics


class ModulationConv2D(Layer):
    """
    Modulation/Demodulation Convolutional layer, including learning rate equilization
    at runtime as per Karras et al. 2017 & 2019. (ProGAN & StyleGan2)

    Inspired by https://github.com/moono/stylegan2-tf-2.x/blob/master/stylegan2/custom_layers.py

    Look at tf-keras branch at
    https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py

    for implementation details.

    """

    def __init__(self, filters, kernel_size, style_fmaps, block, demodulate=False, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.style_fmaps = style_fmaps  # this is z_dim
        self.block = block
        self.demodulate = demodulate

    def build(self, input_shape):
        """
        Input shape is a list of input shapes.

        x_shape = input_shape[0] - the shape of the feature map
        w_shape = input_shape[1] - the shape of the style vector
        """
        x_shape, w_shape = input_shape[0], input_shape[1]

        # print("Build X-shape:", x_shape)

        # Conv Kernel
        self.kernel = self.add_weight("kernel",
                                      shape=(self.kernel_size[0], self.kernel_size[1],
                                             x_shape[-1], self.filters),
                                      initializer=normal(0, 1),
                                      trainable=True
                                      )

        # Equilized learning rate constant
        n = np.product([int(val) for val in x_shape[1:]])
        # He initialisation constant
        self.c = np.sqrt(2 / n)

        # add modulation layer
        self.modulate = DenseEQ(units=x_shape[-1], name=f"modulation_{self.block}")

    def scale_weights(self, style):
        """
        Scales and transforms the weights using the style vector

        B - BATCH
        k - kernel
        I - Input Features
        O - Output Features

        """
        # convolution kernel weights for fused conv
        weight = self.kernel * self.c  # [kkIO]
        weight = weight[np.newaxis]  # [BkkIO]

        # modulation
        style = self.modulate(style)  # [BI] - includes the bias
        weight *= style[:, np.newaxis, np.newaxis, :, np.newaxis]  # [BkkIO]

        # demodulation
        if self.demodulate:
            # demodulate with the L2 Norm of the weights (statistical assumption)
            d = tf.math.rsqrt(tf.reduce_sum(tf.square(weight), axis=[1, 2, 3]) + 1e-8)  # [BO]
            weight *= d[:, np.newaxis, np.newaxis, np.newaxis, :]  # [BkkIO]

        # weight: reshape, prepare for fused operation
        new_weight_shape = [tf.shape(weight)[1], tf.shape(weight)[2], tf.shape(weight)[3], -1]  # [kkI(BO)]
        weight = tf.transpose(weight, [1, 2, 3, 0, 4])  # [kkIBO]
        weight = tf.reshape(weight, shape=new_weight_shape)  # [kkI(BO)]
        return weight

    def call(self, inputs, **kwargs):
        x = inputs[0]
        style = inputs[1]

        # Transform the weights using the style vector
        weights = self.scale_weights(style)

        # Prepare inputs: reshape minibatch to convolution groups
        rows = x.shape[1]
        cols = x.shape[2]
        f = x.shape[3]
        x = tf.reshape(x, [1, -1, rows, cols])

        # Perform convolution
        x = tf.nn.conv2d(x, weights, data_format='NCHW', strides=[1, 1, 1, 1], padding='SAME')

        # x: reshape back to batches
        x = tf.reshape(x, [-1, self.filters, tf.shape(x)[2], tf.shape(x)[3]])

        # x: reshape to [BHWO]
        x = tf.transpose(x, [0, 2, 3, 1])

        x = tf.reshape(x, shape=(-1, rows, cols, f))  # FIXES RESHAPE ISSUE!

        return x

    def get_config(self):
        config = super(ModulationConv2D, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel': self.kernel,
            'style_fmaps': self.style_fmaps,
            'demodulate': self.demodulate,
            'up': self.up,
            'down': self.down
        })
        return config


class Noise(Layer):
    def __init__(self, **kwargs):
        super(Noise, self).__init__(**kwargs)

    def build(self, input_shape):
        w_init = tf.zeros(shape=(), dtype=tf.dtypes.float32)
        self.noise_strength = tf.Variable(w_init, trainable=True, name='w')

    def call(self, inputs, noise=None, training=None, mask=None):
        x_shape = tf.shape(inputs)

        # noise: [1, 1, x_shape[2], x_shape[3]] or None
        if noise is None:
            noise = tf.random.normal(shape=(x_shape[0], 1, x_shape[2], x_shape[3]), dtype=tf.dtypes.float32)

        x = inputs + noise * self.noise_strength
        return x

    def get_config(self):
        config = super(Noise, self).get_config()
        config.update({})
        return config


class Bias(Layer):
    """
    A simple bias layer used in StyleGAN2 after
    the Mod/Demod layer
    """
    def __init__(self, units, lrmul=1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.lrmul = lrmul

    def build(self, input_shape):
        # Conv bias
        self.bias = self.add_weight("bias",
                                    shape=[self.units, ],
                                    initializer=normal(0, 1/self.lrmul)
                                    )

    def call(self, inputs, **kwargs ):
        return inputs + self.bias



