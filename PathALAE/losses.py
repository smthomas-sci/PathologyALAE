import tensorflow as tf
from tensorflow.keras.activations import softplus as f
from tensorflow.keras.losses import binary_crossentropy


class L2(tf.keras.losses.Loss):
    def __init__(self):
        super(L2, self).__init__(reduction=tf.keras.losses.Reduction.NONE, name="l2")

    def call(self, y_true, y_pred):
        loss = (y_true - y_pred) ** 2
        return loss


class DRelative(tf.keras.losses.Loss):
    def __init__(self):
        super(DRelative, self).__init__(reduction=tf.keras.losses.Reduction.NONE, name="dns")

    def call(self, d_real, d_fake):

        logits_diff_real_fake = d_real - tf.reduce_mean(d_fake, axis=0, keepdims=True)
        logits_diff_fake_real = d_fake - tf.reduce_mean(d_real, axis=0, keepdims=True)

        loss_dis_real = tf.reduce_mean(
            binary_crossentropy(y_true=tf.ones_like(d_fake),
                                y_pred=logits_diff_real_fake,
                                from_logits=True))

        loss_dis_fake = tf.reduce_mean(
            binary_crossentropy(y_true=tf.zeros_like(d_fake),
                                y_pred=logits_diff_fake_real,
                                from_logits=True))

        loss = loss_dis_real + loss_dis_fake

        return loss


class GRelative(tf.keras.losses.Loss):
    def __init__(self):
        super(GRelative, self).__init__(reduction=tf.keras.losses.Reduction.NONE, name="gns")

    def call(self, d_real, d_fake):

        logits_diff_real_fake = d_real - tf.reduce_mean(d_fake, axis=0, keepdims=True)
        logits_diff_fake_real = d_fake - tf.reduce_mean(d_real, axis=0, keepdims=True)

        # Generator loss.
        loss_gen_real = tf.reduce_mean(
            binary_crossentropy(y_true=tf.ones_like(d_fake),
                                y_pred=logits_diff_fake_real,
                                from_logits=True))

        loss_gen_fake = tf.reduce_mean(
            binary_crossentropy(y_true=tf.zeros_like(d_fake),
                                y_pred=logits_diff_real_fake,
                                from_logits=True))

        loss = loss_gen_real + loss_gen_fake

        return loss

# Create losses
l2 = L2()
discriminator_relativistic = DRelative()
generator_relativistic = GRelative()
