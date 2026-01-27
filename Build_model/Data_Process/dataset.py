from Build_model.config import *
import tensorflow as tf
import os
import logging
import pickle
from collections import Counter
import re
import string

#chuan hoa van ban
def paragraph_standardization(text):
    return tf.strings.lower(text)


def build_vocab_python(text_dataset, max_tokens):
    logging.info("Building vocabulary using Python (Safe Mode)...")
    vocab_counter = Counter()
    for raw_text in text_dataset:
        text = raw_text.numpy().decode("utf-8")
        text = text.lower()
        tokens = text.split()
        vocab_counter.update(tokens)
    most_common = vocab_counter.most_common(max_tokens)
    vocab_list = [word for word, count in most_common if word not in ["sostoken","eostoken"]]
    vocab_list = vocab_list[:max_tokens-2]
    vocab_list = ["sostoken","eostoken"] +vocab_list
    return vocab_list


# Tao vectorizer
def create_vectorizer(text_dataset, max_tokens, output_sequence_length, vocab_path):
    if os.path.exists(vocab_path):
        try:
            with open(vocab_path, "rb") as f:
                vocab_list = pickle.load(f)
            logging.info(f"Loaded vocab from {vocab_path}")

            vectorizer = tf.keras.layers.TextVectorization(
                max_tokens=max_tokens,
                standardize=paragraph_standardization,
                output_mode="int",
                output_sequence_length=output_sequence_length,
                vocabulary=vocab_list
            )
            return vectorizer
        except Exception as e:
            logging.warning(f"Could not load vocab file: {e}. Rebuilding...")
    vocab_list = build_vocab_python(text_dataset, max_tokens)

    # Lưu vocab_en_to_vi lại
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab_list, f)
    logging.info(f"Saved {len(vocab_list)} vocabs in {vocab_path}")

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        standardize=paragraph_standardization,
        output_mode="int",
        output_sequence_length=output_sequence_length,
        vocabulary=vocab_list
    )
    return vectorizer

def get_vocab_size(vocab_path):
    try:
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
            return len(vocab)+2
    except FileNotFoundError:
        logging.warning(f"File {vocab_path} not found!")
        return VOCAB_SIZE+2

#load dataset
def get_dataset():
    if not os.path.exists(CLEAN_EN_FILE):
        logging.info("File CLEAN_VI_FILE chua ton tai => Chay preprocess.py")
        return None,None,None

    raw_en = tf.data.TextLineDataset(CLEAN_EN_FILE)
    raw_vi = tf.data.TextLineDataset(CLEAN_VI_FILE)

    datasets = tf.data.Dataset.zip((raw_en,raw_vi))

    sample_en = raw_en.take(2000000)
    sample_vi = raw_vi.take(2000000)

    vectorizer_en = create_vectorizer(sample_en,VOCAB_SIZE,MAX_LENGTH,VOCAB_EN_FILE)
    vectorizer_vi = create_vectorizer(sample_vi,VOCAB_SIZE,MAX_LENGTH,VOCAB_VI_FILE)

    def vectorize_text(en,vi):
        en = vectorizer_en(en)
        vi = vectorizer_vi(vi)
        return en,vi

    VAL_SIZE = 20000
    val = datasets.take(VAL_SIZE)
    train = datasets.skip(VAL_SIZE)

    train_dataset = train.map(vectorize_text,num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    val_dataset = val.map(vectorize_text,num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.cache()
    val_dataset = val_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, vectorizer_en, vectorizer_vi

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_ds,val_ds, en_shape, vi_shape = get_dataset()
    if train_ds:
        logging.info("Data Pipeline ready!")

        print("\n--- Checking Train Set ---")
        for batch_en, batch_vi in train_ds.take(1):
            logging.info(f"Train Batch EN Shape: {batch_en.shape}")
            logging.info(f"Train Batch VI Shape: {batch_vi.shape}")

        print("\n--- Checking Validation Set ---")
        for batch_en, batch_vi in val_ds.take(1):
            logging.info(f"Val Batch EN Shape: {batch_en.shape}")
            vocab_en_list = en_shape.get_vocabulary()
            decoded_en = " ".join([vocab_en_list[i] for i in batch_en[0].numpy() if i != 0])
            print(f"Valid example reconstructed: {decoded_en}")