import tensorflow as tf
import time
import logging
import os
import numpy as np
from tensorflow.keras import mixed_precision
from Build_model.config import *
from translation_model import Transformer
from learning_rate import ComputeLR
from Build_model.Data_Process.dataset_vi_en import get_dataset

# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

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
train_dataset,val_dataset,vectorizer_vi, vectorizer_en = get_dataset()
input_vocab_size = len(vectorizer_vi.get_vocabulary())
target_vocab_size = len(vectorizer_en.get_vocabulary())

with tf.device('/GPU:0' if gpus else '/CPU:0'):
    learning_rate = ComputeLR(D_MODEL)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9,beta_2=0.98,epsilon= 1e-9,clipnorm=1.0)
    transformer = Transformer(num_layers = NUM_LAYERS,d_model = D_MODEL,num_heads = NUM_HEADS,dff = DFF,input_vocab_size = input_vocab_size,target_vocab_size = target_vocab_size,dropout_rate = DROPOUT_RATE)

#Tao va load checkpoints
checkpoint_path = "../ModelCheckpoints/VI_EN_Checkpoint"
checkpnt = tf.train.Checkpoint(transformer=transformer,optimizer=optimizer)
checkpnt_manager = tf.train.CheckpointManager(checkpnt,checkpoint_path,max_to_keep=5)

best_val_loss = float("inf")

if checkpnt_manager.latest_checkpoint:
    checkpnt.restore(checkpnt_manager.latest_checkpoint)
    logging.info("Restore from {}".format(checkpnt_manager.latest_checkpoint))

#Loss function
loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")

val_loss = tf.keras.metrics.Mean(name="val_loss")
val_accuracy = tf.keras.metrics.Mean(name="val_accuracy")
def lossfunc(y,pred):
    mask = tf.math.logical_not(tf.math.equal(y,0))

    loss_ = loss_obj(y,pred)

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

@tf.function
def train_step(inp,tar):
    tar_inp = tar[:,:-1]
    tar_y = tar[:,1:]

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp=inp,tar=tar_inp, training=True)
        loss_value = lossfunc(tar_y,predictions)
    gradients = tape.gradient(loss_value, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients,transformer.trainable_variables))

    batch_accuracy = accuracyfunc(tar_y,predictions)

    train_loss(loss_value)
    train_accuracy(batch_accuracy)

@tf.function
def val_step(inp,tar):
    tar_inp = tar[:, :-1]
    tar_y = tar[:, 1:]

    predictions, _ = transformer(inp=inp, tar=tar_inp, training=False)

    v_loss = lossfunc(tar_y, predictions)
    v_acc = accuracyfunc(tar_y, predictions)

    val_loss(v_loss)
    val_accuracy(v_acc)

epochs = 30

for epoch in range(epochs):
    start = time.time()
    train_loss.reset_state()
    train_accuracy.reset_state()
    val_loss.reset_state()
    val_accuracy.reset_state()
    for (batch, (inpepoch, tarepoch)) in enumerate(train_dataset):
        train_step(inp=inpepoch, tar=tarepoch)
        if batch % 200 == 0:
            print(f"Epoch{epoch + 1} Train_Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}")
    print("Validation")
    for (inp_val, out_val) in val_dataset:
        val_step(inp_val, out_val)
    print("Validation Done!")
    current_val_loss = val_loss.result()
    print(f"Result of Epoch: {epoch + 1}")
    print(f"TrainLoss {train_loss.result():.4f} TrainAccuracy {train_accuracy.result():.4f}")
    print(f"ValLoss {current_val_loss:.4f} ValAccuracy {val_accuracy.result():.4f}")
    print(f"Time taken for 1 epoch: {time.time() - start:.2f} secs\n")

    if current_val_loss < best_val_loss:
        print(f"Val_loss spread: {current_val_loss - best_val_loss:.4f}")
        best_val_loss = current_val_loss
        checkpnt_manager.save()
    else:
        print("Val_loss did not improve!")
print("Training process complete!")