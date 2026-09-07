import tensorflow as tf
from tensorflow.keras import layers, regularizers

class BiDAFAttention(layers.Layer):
    """Bi-Directional Attention Flow (Seo et al. 2017).
    inputs = [context H (B,T,2d), question U (B,J,2d)]  ->  G (B,T,8d)"""

    def build(self, input_shape):
        d2 = input_shape[0][-1]                      # 2*HIDDEN
        reg = regularizers.l2(1e-4)
        self.w_h  = self.add_weight((d2, 1), initializer="glorot_uniform", name="w_h", regularizer=reg)
        self.w_u  = self.add_weight((d2, 1), initializer="glorot_uniform", name="w_u", regularizer=reg)
        self.w_hu = self.add_weight((d2,),   initializer="glorot_uniform", name="w_hu", regularizer=reg)
        super().build(input_shape)

    def call(self, inputs, mask=None):
        H, U = inputs
        T = tf.shape(H)[1]

        # similarity matrix S (B,T,J), efficient decomposed form
        term_h  = tf.matmul(H, self.w_h)                              # (B,T,1)
        term_u  = tf.transpose(tf.matmul(U, self.w_u), [0, 2, 1])     # (B,1,J)
        term_hu = tf.matmul(H * self.w_hu, U, transpose_b=True)       # (B,T,J)
        S = term_h + term_u + term_hu                                 # (B,T,J)

        # mask out padded QUESTION positions before softmax
        if mask is not None and mask[1] is not None:
            qm = tf.cast(mask[1], S.dtype)[:, tf.newaxis, :]          # (B,1,J)
            S = S + (1.0 - qm) * -1e9

        # Context-to-Query: each context word attends over the question
        a = tf.nn.softmax(S, axis=-1)
        U_tilde = tf.matmul(a, U)                                     # (B,T,2d)

        # Query-to-Context: the context words most relevant to the question
        b = tf.nn.softmax(tf.reduce_max(S, axis=-1), axis=-1)        # (B,T)
        H_tilde = tf.tile(tf.matmul(b[:, tf.newaxis, :], H), [1, T, 1])  # (B,T,2d)

        # fuse into G (B,T,8d)
        return tf.concat([H, U_tilde, H * U_tilde, H * H_tilde], axis=-1)

    def compute_mask(self, inputs, mask=None):
        return mask[0] if mask is not None else None   # pass context mask downstream