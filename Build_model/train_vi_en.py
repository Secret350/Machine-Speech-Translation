import tensorflow as tf
import time
import logging
import os
from tensorflow.keras import mixed_precision
import config
from translation_model import Transformer
from learning_rate import ComputeLR
from dataset_vi_en import get_dataset,get_vocab_size

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f">>> Đã tìm thấy {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        print(f">>> Đang sử dụng GPU: {tf.config.list_physical_devices('GPU')[0].name}")
    except RuntimeError as e:
        print(f"Lỗi cấu hình GPU: {e}")
else:
    print(">>> CẢNH BÁO: Không tìm thấy GPU, hệ thống sẽ chạy trên CPU.")

#Lay data va dinh nghia mo hinh, setup mo hình
train_dataset,vectorizer_vi, vectorizer_en = get_dataset()
input_vocab_size = len(vectorizer_vi.get_vocabulary())
target_vocab_size = len(vectorizer_en.get_vocabulary())

with tf.device('/GPU:0' if gpus else '/CPU:0'):
    learning_rate = ComputeLR(config.D_MODEL)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9,beta_2=0.98,epsilon= 1e-9,clipnorm=1.0)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    transformer = Transformer(num_layers = config.NUM_LAYERS,d_model = config.D_MODEL,num_heads = config.NUM_HEADS,dff = config.DFF,input_vocab_size = input_vocab_size,target_vocab_size = target_vocab_size,dropout_rate = config.DROPOUT_RATE)

#Tao va load checkpoints
checkpoint_path = "./Checkpoint_Vi-to-En/Train"
checkpnt = tf.train.Checkpoint(transformer=transformer,optimizer=optimizer)
checkpnt_manager = tf.train.CheckpointManager(checkpnt,checkpoint_path,max_to_keep=5)

if checkpnt_manager.latest_checkpoint:
    checkpnt.restore(checkpnt_manager.latest_checkpoint)
    logging.info("Restore from {}".format(checkpnt_manager.latest_checkpoint))

#Loss function
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

def lossfunc(y,pred):
    mask = tf.math.logical_not(tf.math.equal(y,0))

    loss_ = loss(y,pred)

    mask = tf.cast(mask, dtype=loss_.dtype)

    loss_ *= mask

    num_active_elements = tf.reduce_sum(mask)
    return tf.reduce_sum(loss_) / tf.maximum(num_active_elements, 1.0)

def accuracyfunc(y,pred):
    accuracies = tf.equal(y, tf.argmax(pred,axis=2))

    mask = tf.math.logical_not(tf.math.equal(y,0))

    accuracies = tf.math.logical_and(mask,accuracies)

    accuracies = tf.cast(accuracies,dtype=tf.float32)

    mask = tf.cast(mask, dtype=tf.float32)

    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

train_loss = tf.keras.metrics.Mean(name= "train_loss")
train_accuracy = tf.keras.metrics.Mean(name= "train_accuracy")

@tf.function
def train_step(inp,tar):
    tar_inp = tar[:,:-1]
    tar_y = tar[:,1:]

    tf.debugging.assert_less(
        tf.reduce_max(tar_y),
        tf.cast(target_vocab_size, tf.int64),
        message="ER: Co Token ID trong du lieu lon hon kich thuoc Output Layer cua Model!"
    )

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp=inp,tar=tar_inp, training=True)
        loss_tape = lossfunc(tar_y,predictions)
        scaled_loss = optimizer.get_scaled_loss(loss_tape)
    scaled_gradients = tape.gradient(scaled_loss,transformer.trainable_variables)
    gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    optimizer.apply_gradients(zip(gradients,transformer.trainable_variables))

    batch_accuracy = accuracyfunc(tar_y,predictions)

    train_loss(loss_tape)
    train_accuracy(batch_accuracy)

epochs = 20

for epoch in range(epochs):
    start = time.time()
    train_loss.reset_state()
    train_accuracy.reset_state()
    for (batch,(inpepoch,tarepoch)) in enumerate(train_dataset):
        train_step(inp=inpepoch,tar=tarepoch)
        if batch%100 == 0:
            print(f"Epoch{epoch+1} Batch {batch}  Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}")

    checkpoint_path = checkpnt_manager.save()
    print(f"Saving checkpoint for epoch {epoch+1}")
    print(f"Epoch {epoch+1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}")
    print(f"Time taken for 1 epoch: {time.time()-start:.2f} secs\n")