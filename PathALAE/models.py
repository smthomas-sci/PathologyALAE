from PathALAE.blocks import *
from PathALAE.losses import *
from PathALAE.layers import *

import numpy as np


# Mapping Network Residual
class MappingResidual(Model):
    """
    Builds residual mapping network!
    """
    def __init__(self, z_dim, n_layers=5, lrmul=0.01, **kwargs):
        super(MappingResidual, self).__init__(**kwargs)
        self.z_dim = z_dim
        self.n_layers = n_layers

        for i in range(self.n_layers):
            setattr(self, f"f_{i+1}", DenseEQ(units=z_dim, name=f"F_dense_{i+1}", lrmul=lrmul))
            setattr(self, f"f_act_{i+1}", LeakyReLU(0.2, name=f"F_act_{i+1}"))

    def call(self, inputs, **kwargs):

        x = inputs

        for i in range(self.n_layers):
            # Input is last output
            inputs = x
            # Get layers
            layer = getattr(self, f"f_{i+1}")
            act = getattr(self, f"f_act_{i+1}")

            # transform
            x = layer(x)
            x = act(x)
            # Make residual
            x = x + inputs

        return x


def build_demod_generator(z_dim, base_dim, image_dim, filters, attention_level):
    # Define inputs
    g_style = Input(shape=(z_dim,))
    g_noise = Input(shape=(image_dim, image_dim, 1))

    x = g_style

    # Layer 1
    x = DenseEQ(units=1024)(x)
    x = LeakyReLU(0.2)(x)

    # Layer 2
    x = DenseEQ(units=np.product(base_dim))(x)
    x = LeakyReLU(0.2)(x)

    # Reshape
    x = Reshape(base_dim, name="Gen_Base_Reshape")(x)

    # Add blocks
    for b, f in enumerate(filters):

        # exit condition!
        if b == len(filters) - 2:
            break

        # Attention!
        if b == attention_level:
            x = Attention(ch=f, name=f"Block_{b + 1}_Attention")(x)

        # Residual + Upsample
        x = ResidualBlockDeMod(filters=f, block=b, z_dim=z_dim, name=f"GenBlock_{b + 1}")([x, g_style])
        x = UpSampling2D(name=f"GenUP_{b + 1}", interpolation="bilinear")(x)
        x = Conv2DEQ(filters=filters[b + 1], kernel_size=(3, 3), padding="same", name=f"GenBlock_{b + 1}_UpConv")(x)
        x = LeakyReLU(0.2)(x)

    # Convert to RGB
    x = tRGB(block=len(filters) - 1)(x)
    # g_out = tf.keras.activations.sigmoid(x)
    g_out = x

    # Build model
    return Model(inputs=[g_style, g_noise], outputs=[g_out], name="G")


def build_adain_generator(z_dim, image_dim, base_dim, filters, attention_level):
    """

    :param z_dim: the size of the latent dimension, e.g. 200
    :param image_dim: the maximum size eg. 256 for 256x256 pixel images.
    :param base_dim: (4, 4, 256) - must be at 4x4 level (fix later)
    :param filters: list of filers at each level
    :param attention_level: the level to include an attention block.
    :return: the generator model
    """
    # Define inputs
    g_style = Input(shape=(z_dim,))
    g_noise = Input(shape=(image_dim, image_dim, 1))

    x = g_style

    # Layer 1
    x = DenseEQ(units=1024)(x)
    x = LeakyReLU(0.2)(x)

    # Layer 2
    x = DenseEQ(units=np.product(base_dim))(x)
    x = LeakyReLU(0.2)(x)

    # Reshape
    x = Reshape(base_dim, name="Gen_Base_Reshape")(x)

    # Add blocks
    for b, f in enumerate(filters):

        # exit condition!
        if b == len(filters)-2:
            break

        # Attention!
        if b == attention_level:
            x = Attention(ch=f, name=f"Block_{b+1}_Attention")(x)

        # Residual + Upsample
        x = ResidualBlockAdaIN(filters=f, block=b+1, name=f"GenBlock_{b+1}")([x, g_noise, g_style])
        x = UpSampling2D(name=f"GenUP_{b+1}", interpolation="bilinear")(x)
        x = Conv2DEQ(filters=filters[b+1], kernel_size=(3, 3), padding="same", name=f"GenBlock_{b+1}_UpConv")(x)
        x = LeakyReLU(0.2)(x)

    # Convert to RGB
    x = tRGB(block=len(filters)-1)(x)
    #g_out = tf.keras.activations.sigmoid(x)
    g_out = x

    # Build model
    return Model(inputs=[g_style, g_noise], outputs=[g_out], name="G")


def build_style_encoder(z_dim, image_dim, filters, attention_level):
    """

    :param z_dim:
    :param image_dim:
    :param filters:
    :param attention_level:
    :return:
    """
    # Define inputs
    e_in = Input(shape=(image_dim, image_dim, 3))

    x = e_in
    ws = []

    # First convolution
    x = Conv2DEQ(filters=filters[::-1][0],
                 kernel_size=(3, 3),
                 padding="same",
                 name=f"EncBlock_0_Conv")(x)
    x = LeakyReLU(0.2, name="EncBlock_0_act")(x)

    for b, f in enumerate(filters[::-1]):

        # exit condition
        if b == len(filters) - 1:
            break

        # Attention!
        if b == attention_level:
            x = Attention(ch=f, name=f"Block_{b + 1}_Attention")(x)

        x, w1, w2 = ResidualBlockInNorm(filters=f,
                                        block=b,
                                        z_dim=z_dim,
                                        name=f"E_block_{b+1}_encoder")(x)
        # Down sample
        x = AveragePooling2D(name=f"E_block_{b}_DownSample")(x)
        x = Conv2DEQ(filters=filters[::-1][b + 1],
                     kernel_size=(3, 3),
                     padding="same",
                     name=f"EncBlock_{b + 1}_UpConv")(x)
        x = LeakyReLU(0.2, name=f"EncBlock_down_act_{b}")(x)

        ws.extend([w1, w2])

    # Sum all descriptions
    e_out = Add(name="E_Sum_W")(ws)

    return Model(inputs=[e_in], outputs=[e_out], name=f"E")


def build_encoder(z_dim, image_dim, filters, attention_level):
    """

    :param z_dim:
    :param image_dim:
    :param filters:
    :param attention_level:
    :return:
    """
    # Define inputs
    e_in = Input(shape=(image_dim, image_dim, 3))

    x = e_in

    # First convolution
    x = Conv2DEQ(filters=filters[::-1][0],
                 kernel_size=(3, 3),
                 padding="same",
                 name=f"EncBlock_0_Conv")(x)
    x = LeakyReLU(0.2, name="EncBlock_0_act")(x)

    for b, f in enumerate(filters[::-1]):

        # exit condition
        if b == len(filters) - 1:
            break

        # Attention!
        if b == attention_level:
            x = Attention(ch=f, name=f"Block_{b + 1}_Attention")(x)

        x = ResidualBlock(filters=f,
                          block=b,
                          name=f"E_block_{b+1}_encoder")(x)

        if x.shape[1] != 8: # FIX THIS - Needs to be controlled by base_dim
            # Down sample
            x = AveragePooling2D(name=f"E_block_{b}_DownSample")(x)
        x = Conv2DEQ(filters=filters[::-1][b + 1],
                     kernel_size=(3, 3),
                     padding="same",
                     name=f"EncBlock_{b + 1}_UpConv")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2, name=f"EncBlock_down_act_{b}")(x)

    x = Flatten(name="E_Flatten")(x)

    x = DenseEQ(units=1024, name=f"E_bottleneck")(x)
    x = LeakyReLU(0.2, name=f"E_bottleneck_act_1")(x)

    # Map to W space
    e_out = DenseEQ(units=z_dim, name=f"E_latent_return")(x)

    return Model(inputs=[e_in], outputs=[e_out], name=f"E")


class DiscriminatorResidual(Model):
    """
    Builds residual mapping network!
    """
    def __init__(self, z_dim, n_layers=5, lrmul=0.01, **kwargs):
        super(DiscriminatorResidual, self).__init__(**kwargs)
        self.z_dim = z_dim
        self.n_layers = n_layers

        for i in range(self.n_layers):
            setattr(self, f"f_{i+1}", DenseEQ(units=z_dim, name=f"F_dense_{i+1}", lrmul=lrmul))
            setattr(self, f"f_act_{i+1}", LeakyReLU(0.2, name=f"F_act_{i+1}"))

        setattr(self, f"f_final", DenseEQ(units=1, name=f"F_dense_final", lrmul=lrmul))

    def call(self, inputs, **kwargs):

        x = inputs

        for i in range(self.n_layers):
            # Input is last output
            inputs = x
            # Get layers
            layer = getattr(self, f"f_{i+1}")
            act = getattr(self, f"f_act_{i+1}")

            # transform
            x = layer(x)
            x = act(x)
            # Make residual
            x = x + inputs

        # Final
        layer = getattr(self, f"f_final")
        x = layer(x)
        # No activation - apply in loss

        return x


class ALAE(Model):
    """
    An Adversarial Latent Autoencoder Model (ALAE), self-contained for
    training at each block size.
    """
    def __init__(self, x_dim, z_dim, f_model, g_model, e_model, d_model, style_mix_step=16):
        super(ALAE, self).__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.F = f_model
        self.G = g_model
        self.E = e_model
        self.D = d_model
        self.levels = int(np.log2(self.x_dim/2))  # the number of blocks
        self.style_mix_step = style_mix_step

        # Composite models
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()
        self.reciprocal = self.build_reciprocal()
        self.inference = self.build_inference()
        # self.styleMixer = self.build_styleMixer() TO DO

    def build_discriminator(self):
        discriminator_in = Input(shape=(self.x_dim, self.x_dim, 3))
        discriminator_out = self.D(self.E(discriminator_in))
        return Model(inputs=[discriminator_in], outputs=[discriminator_out], name="discriminator")

    def build_generator(self):
        z_input = Input(shape=(self.z_dim,))
        noise_input = Input(shape=(self.x_dim, self.x_dim, 1))
        generator_ins = [z_input, noise_input]

        # Map z -> w
        w = self.F(z_input)
        #x = self.G([w]*self.levels + [noise_input])
        x = self.G([w] + [noise_input])
        generator_out = x
        return Model(inputs=generator_ins, outputs=[generator_out], name="generator")

    def build_reciprocal(self):
        w_in = Input(shape=(self.z_dim,))  # W is input
        noise_input = Input(shape=(self.x_dim, self.x_dim, 1))
        reciprocal_ins = [w_in, noise_input]

        #g_ins = [w_in]*self.levels + [noise_input]
        g_ins = [w_in] + [noise_input]
        reciprocal_out = self.E(self.G(g_ins))
        return Model(inputs=reciprocal_ins, outputs=[reciprocal_out], name="reciprocal")

    def build_inference(self):
        inference_in = Input(shape=(self.x_dim, self.x_dim, 3), name="inference_input")
        noise_input = Input(shape=(self.x_dim, self.x_dim, 1))

        inference_ins = [inference_in, noise_input]

        w = self.E(inference_in)
        #g_ins = [w]*self.levels + [noise_input]
        g_ins = [w] + [noise_input]
        inference_out = self.G(g_ins)
        return Model(inputs=inference_ins, outputs=[inference_out], name="inference")

    def compile(self, d_optimizer, g_optimizer,  r_optimizer, γ=10, alpha_step=None, **kwargs):
        """
        Overrides the compile step. If it is a merge model,
        then the Fade layers are initialised and alphas are
        created.
        """
        super(ALAE, self).compile()
        self.optimizer_d = d_optimizer
        self.optimizer_g = g_optimizer
        self.optimizer_r = r_optimizer
        self.γ = γ

        # get trainable params
        self.θ_F = self.F.trainable_weights
        self.θ_G = self.G.trainable_weights
        self.θ_E = self.E.trainable_weights
        self.θ_D = self.D.trainable_weights

        # Create loss trackers
        self.d_loss_tracker = tf.keras.metrics.Mean(name="loss_d")
        self.g_loss_tracker = tf.keras.metrics.Mean(name="loss_g")
        self.r_loss_tracker = tf.keras.metrics.Mean(name="loss_r")
        self.gp_loss_tracker = tf.keras.metrics.Mean(name="loss_r1")

        # Create internal step tracker
        self.step = 0

    # def style_mixing_regularization(self, batch_size, noise, real_pred):
    #     # Reset step
    #     self.step *= 0
    #
    #     # samples from prior N(0, 1)
    #     z1 = tf.random.normal(shape=(batch_size, self.z_dim))
    #     z2 = tf.random.normal(shape=(batch_size, self.z_dim))
    #
    #     # Get ws
    #     w1 = self.F(z1)
    #     w2 = self.F(z2)
    #
    #     # Get random position
    #     pos = tf.random.uniform(shape=[], minval=0, maxval=self.levels, dtype="int32")
    #
    #     # Compute loss and apply gradients
    #     with tf.GradientTape() as tape:
    #         x = self.G([w1]*pos + [w2]*((self.levels-1) - pos) + [noise])
    #         fake_pred = self.discriminator(x)
    #         loss_g_style_mix = generator_relativistic(real_pred, fake_pred)
    #     gradients = tape.gradient(loss_g_style_mix, self.θ_F + self.θ_G)
    #     self.optimizer_g.apply_gradients(zip(gradients, self.θ_F + self.θ_G))

    def train_step(self, batch):
        """
        Custom training step - follows algorithm of ALAE e.g. Step I,II & III
        :param batch:
        :return: losses
        """
        batch_size = batch[0].shape[0]

        if not batch_size:
            batch_size = 1

        # -------------------------------#
        # Step I - Update Discriminator  #
        # -------------------------------#

        # Random mini-batch from data set
        x_real, noise = batch

        batch_size = tf.shape(x_real)[0]

        # samples from prior N(0, 1)
        z = tf.random.normal(shape=(batch_size, self.z_dim))
        # generate fake images
        x_fake = self.generator([z, noise])

        # Compute loss and apply gradients
        with tf.GradientTape() as tape:

            fake_pred = self.discriminator(x_fake)

            real_pred = self.discriminator(x_real)

            loss_d = discriminator_relativistic(real_pred, fake_pred)

            # Add the R1 term
            if self.γ > 0:

                with tf.GradientTape() as r1_tape:
                    r1_tape.watch(x_real)
                    # 1. Get the discriminator output for real images
                    pred = self.discriminator(x_real)

                # 2. Calculate the gradients w.r.t to the real images.
                grads = r1_tape.gradient(pred, [x_real])[0]

                # 3. Calculate the squared norm of the gradients
                r1_penalty = tf.reduce_sum(tf.square(grads))
                loss_d += self.γ / 2 * r1_penalty

        gradients = tape.gradient(loss_d, self.θ_E + self.θ_D)
        self.optimizer_d.apply_gradients(zip(gradients, self.θ_E + self.θ_D))

        # ----------------------------#
        #  Step II - Update Generator #
        # ----------------------------#

        # samples from prior N(0, 1)
        z = tf.random.normal(shape=(batch_size, self.z_dim))
        # Compute loss and apply gradients
        with tf.GradientTape() as tape:

            fake_pred = self.discriminator(self.generator([z, noise]))

            loss_g = generator_relativistic(real_pred, fake_pred)

        gradients = tape.gradient(loss_g, self.θ_F + self.θ_G)
        self.optimizer_g.apply_gradients(zip(gradients, self.θ_F + self.θ_G))

        # -------  Style Mixing ------- #
        self.step += 1
        # if self.step % self.style_mix_step == 0:
        #     print("Style mixing...")
        #     self.style_mixing_regularization(batch_size, noise, real_pred)
        # ------------------------------#

        # ------------------------------#
        #  Step III - Update Reciprocal #
        # ------------------------------#

        # samples from prior N(0, 1)
        z = tf.random.normal(shape=(batch_size, self.z_dim))
        # Get w
        w = self.F(z)
        # Compute loss and apply gradients
        with tf.GradientTape() as tape:

            w_pred = self.reciprocal([w, noise])

            loss_r = l2(w, w_pred)

        gradients = tape.gradient(loss_r, self.θ_G + self.θ_E)
        self.optimizer_r.apply_gradients(zip(gradients, self.θ_G + self.θ_E))

        if self.γ == 0:
             r1_penalty = 0.0

        # Update loss trackers
        self.d_loss_tracker.update_state(loss_d)
        self.g_loss_tracker.update_state(loss_g)
        self.r_loss_tracker.update_state(loss_r)
        self.gp_loss_tracker.update_state(r1_penalty)

        return {"loss_d": self.d_loss_tracker.result(),
                "loss_g": self.g_loss_tracker.result(),
                "loss_r": self.r_loss_tracker.result(),
                "loss_gp": self.gp_loss_tracker.result()}

    def call(self, inputs):
        return inputs

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker, self.r_loss_tracker, self.gp_loss_tracker]

