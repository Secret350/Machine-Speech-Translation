import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout,Layer
from transformer_layers import EncoderLayer,DecoderLayer,PositionalEmbedding

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq,0),tf.float32)
    return seq[:,tf.newaxis,tf.newaxis,:]

#taoj look_ahead_mask de che di tu tiep theo, tranh mo hinh nhin trom tu tiep theo truoc khi du doan
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size,size)),-1 , 0)
    return mask

class Encoder(Layer):
    def __init__(self,num_layers, d_model, num_heads,dff,vocab_size, dropout_rate=0.1):
        super(Encoder,self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.positional_embedding = PositionalEmbedding(vocab_size,d_model)

        self.encoding = [EncoderLayer(d_model,num_heads,dff,dropout_rate) for _ in range(num_layers)]
        self.dropout = Dropout(dropout_rate)

    def call(self, x, training=None, mask=None):
        x = self.positional_embedding(x)
        x = self.dropout(x, training=training)

        for i in range (self.num_layers):
            x = self.encoding[i](x,training=training,mask=mask)
        return x

class Decoder(Layer):
    def __init__(self,num_layers,d_model,num_heads, dff,vocab_size, dropout_rate=0.1):
        super(Decoder,self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.positional_embedding = PositionalEmbedding(vocab_size,d_model)

        self.decoding = [DecoderLayer(d_model,num_heads,dff,dropout_rate) for _ in range (num_layers)]
        self.dropout = Dropout(dropout_rate)

    def call(self,x,encoder_output=None,training=None,look_ahead_mask=None,padding_mask=None):
        x = self.positional_embedding(x)
        x = self.dropout(x,training=training)

        attention_weight = {}

        for i in range (self.num_layers):
            x , attention_block_1,attention_block_2 = self.decoding[i](x,encoder_output=encoder_output,training=training,look_ahead_mask=look_ahead_mask,padding_mask=padding_mask)
            attention_weight[f"decoder_layer{i + 1}_block_1"] = attention_block_1
            attention_weight[f"decoder_layer{i + 1}_block_2"] = attention_block_2
        return x, attention_weight


def create_masks(inp, tar):
    encoder_padding_mask = create_padding_mask(inp)
    decoding_padding_mask = create_padding_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    decoder_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(decoder_target_padding_mask,look_ahead_mask)

    return combined_mask,encoder_padding_mask,decoding_padding_mask


class Transformer(Model):
    def __init__(self,num_layers,d_model,num_heads,dff,input_vocab_size,target_vocab_size,dropout_rate=0.1):
        super(Transformer,self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads,dff,input_vocab_size, dropout_rate=dropout_rate)
        self.decoder = Decoder(num_layers,d_model,num_heads, dff,target_vocab_size, dropout_rate=dropout_rate)
        self.linear_layer = Dense(target_vocab_size,dtype="float32")

    def call(self, inp,tar=None,training=None):
        combined_mask,encoder_padding_mask,decoding_padding_mask = create_masks(inp,tar)

        encoder_output = self.encoder(x=inp,training=training,mask=encoder_padding_mask)

        decoder_output, attention_weight = self.decoder(x=tar,encoder_output=encoder_output,training=training,look_ahead_mask=combined_mask,padding_mask=decoding_padding_mask)

        final_output = self.linear_layer(decoder_output)

        return final_output,attention_weight