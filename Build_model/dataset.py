from config import *
import tensorflow as tf
import os
import logging
import pickle
from collections import Counter

#chuan hoa van ban
def paragraph_standardization(text):
    text = tf.strings.lower(text)
    return text


def build_vocab_python(text_dataset, max_tokens):
    logging.info("Building vocabulary using Python (Safe Mode)...")
    vocab_counter = Counter()
    for raw_text in text_dataset:
        text = raw_text.numpy().decode("utf-8")
        text = text.lower()
        tokens = text.split()
        vocab_counter.update(tokens)
    most_common = vocab_counter.most_common(max_tokens - 2)
    vocab_list = [word for word, count in most_common]
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

    # Lưu vocab lại
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

    preprocess_dataset = datasets.map(vectorize_text,num_parallel_calls=tf.data.AUTOTUNE)
    preprocess_dataset = preprocess_dataset.cache()
    preprocess_dataset = preprocess_dataset.shuffle(BUFFER_SIZE)
    preprocess_dataset = preprocess_dataset.batch(BATCH_SIZE)
    preprocess_dataset = preprocess_dataset.prefetch(tf.data.AUTOTUNE)

    return preprocess_dataset, vectorizer_en, vectorizer_vi

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ds, en_shape, vi_shape = get_dataset()
    if ds:
        logging.info("Data Pipline ready!")
        vocab_en_list = en_shape.get_vocabulary()
        en_size = get_vocab_size(VOCAB_EN_FILE)
        vi_size = get_vocab_size(VOCAB_VI_FILE)
        logging.info(f"Vocab Size loaded from file: EN={en_size}, VI={vi_size}")
        for batch_en, batch_vi in ds.take(1):
            logging.info(f"Shape batch English: {batch_en.shape}")
            logging.info(f"Shape batch Vietnamese: {batch_vi.shape}")
            vocab_en_list = en_shape.get_vocabulary()
            decoded_en = " ".join([vocab_en_list[i] for i in batch_en[0].numpy() if i != 0])
            print(f"English example reconstructed: {decoded_en}")