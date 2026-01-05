import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense,Layer, LayerNormalization,Dropout,Embedding

#Xay dung Positon Encoding

def get_angles(pos,i,d_model):
    angles_rates = 1/np.power(10000,((2*i)//np.float32(d_model)))
    return pos*angles_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(Layer):
    def __init__(self, vocab_size, d_model):
        super(PositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.embedding = Embedding(vocab_size, d_model,mask_zero=False)
        self.pos_encoding = positional_encoding(2048, d_model)
    def compute_mask(self, inputs, mask=None):
        return None
    def call(self, inputs, **kwargs):
        seq_len = tf.shape(inputs)[1]

        inputs = self.embedding(inputs)

        inputs = tf.cast(inputs, tf.float32)
        inputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        inputs += self.pos_encoding[:, :seq_len, :]

        return inputs

#Xay dung multihead_attention
def scales_dot_product_attention(q,k,v,mask):

    dot_qk = tf.matmul(q,k,transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1],tf.float32)

    scale_attention_logit = tf.cast(dot_qk,tf.float32)/tf.sqrt(dk)

    if mask is not None:
        mask = tf.cast(mask, tf.float32)
        scale_attention_logit += (mask* -1e9)

    softmax_attention = tf.nn.softmax(scale_attention_logit,axis=-1)
    softmax_attention = tf.cast(softmax_attention, dtype=v.dtype)
    output = tf.matmul(softmax_attention,v)

    return output , softmax_attention

class MultiheadAttention(Layer):
    def __init__(self,d_model,num_heads):
        super(MultiheadAttention,self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0,"d_model phai bang #heads"
        
        self.depth = d_model//num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self,x,batch_size):
        x = tf.reshape(x,(batch_size,-1,self.num_heads,self.depth))

        return tf.transpose(x, perm=[0,2,1,3])
    def call(self, q=None, k=None, v=None, mask=None):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q,batch_size)
        k = self.split_heads(k,batch_size)
        v = self.split_heads(v,batch_size)

        scaled_attention, attention_weight = scales_dot_product_attention(q,k,v,mask)
        scaled_attention = tf.transpose(scaled_attention,perm=[0,2,1,3])

        concat_attention = tf.reshape(scaled_attention,(batch_size,-1,self.d_model))
        output = self.dense(concat_attention)

        return output, attention_weight

#Xay dung Feed Forward Neural Network

class FeedForwardNeuralNetwork(Layer):
    def __init__(self, d_model, dff):
        super(FeedForwardNeuralNetwork, self).__init__()
        self.dense1 = Dense(dff, activation='relu')
        self.dense2 = Dense(d_model)

    def call(self, inputs, **kwargs):
        inputs = self.dense1(inputs)
        inputs = self.dense2(inputs)
        return inputs

class EncoderLayer(Layer):
    def __init__(self,d_model,num_heads,dff,dropout_rate=0.1):
        super(EncoderLayer,self).__init__()
        
        #cac ham multiheadattention, add&norm, feedforwardneuralnetwork
        self.MultiHead = MultiheadAttention(d_model,num_heads)
        self.FFNN = FeedForwardNeuralNetwork(d_model,dff)
        self.Add_Norm1 = LayerNormalization(epsilon=1e-6)
        self.Add_Norm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    def call(self,x=None,training=None,mask=None):
        #Xay dung cau truc block encoder multihead -> dropout -> add&norm1
        attention_output,_ = self.MultiHead(x,x,x,mask)
        attention_output = self.dropout1(attention_output,training=training)
        encoder_output1 = self.Add_Norm1(x+attention_output)
        # feedforwardneuralnetwork -> dropout -> add&norm -> encoder_output
        ffnn_output = self.FFNN(encoder_output1)
        ffnn_output = self.dropout2(ffnn_output,training=training)
        out_encoder = self.Add_Norm2(encoder_output1+ffnn_output)
        
        return out_encoder

#Xay dung block decoder
class DecoderLayer(Layer):
    def __init__(self,d_model,num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        
        self.MultiHead1 = MultiheadAttention(d_model,num_heads)
        self.MultiHead2 = MultiheadAttention(d_model,num_heads)
        self.FFNN = FeedForwardNeuralNetwork(d_model,dff)

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)

        self.Add_Norm1 = LayerNormalization(epsilon=1e-6)
        self.Add_Norm2 = LayerNormalization(epsilon=1e-6)
        self.Add_Norm3 = LayerNormalization(epsilon=1e-6)

    def call(self,x=None,encoder_output=None,training=None,look_ahead_mask=None, padding_mask=None):
        Attention1, attention_weight_block1= self.MultiHead1(x,x,x,look_ahead_mask)
        Attention1 = self.dropout1(Attention1,training=training)
        decoder_output1 = self.Add_Norm1(Attention1+x)

        Attention2, attention_weight_block2 = self.MultiHead2(q=decoder_output1,k=encoder_output,v=encoder_output,mask=padding_mask)
        Attention2 = self.dropout2(Attention2,training=training)
        decoder_output2 = self.Add_Norm2(Attention2+decoder_output1)

        ffnn_output = self.FFNN(decoder_output2)
        ffnn_output = self.dropout3(ffnn_output,training=training)
        decoder_output3 = self.Add_Norm3(ffnn_output+decoder_output2)

        return decoder_output3,attention_weight_block1,attention_weight_block2