from config import *
import tensorflow as tf
import os
import logging
import pickle

#chuan hoa van ban
def paragraph_standardization(text):
    text = tf.strings.lower(text)
    return text

#tao vectorizer
def create_vectorizer (text_dataset, max_tokens, output_sequence_length, vocab_path):
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        standardize=paragraph_standardization,
        output_mode="int",
        output_sequence_length=output_sequence_length
    )
    logging.info("Adapting vectorizer...")
    vectorizer.adapt(text_dataset)

    if vocab_path:
        vocab = vectorizer.get_vocabulary()
        with open(vocab_path,"wb") as f:
            pickle.dump(vocab,f)
        logging.info(f"Saved {len(vocab)} vocabs in {vocab_path}")

    return vectorizer

#load dataset
def get_dataset():
    if not os.path.exists(CLEAN_EN_FILE):
        logging.info("File CLEAN_VI_FILE chua ton tai => Chay preprocess.py")
        return None,None,None

    raw_en = tf.data.TextLineDataset(CLEAN_EN_FILE)
    raw_vi = tf.data.TextLineDataset(CLEAN_VI_FILE)

    datasets = tf.data.Dataset.zip((raw_en,raw_vi))

    sample_en = raw_en.take(10000)
    sample_vi = raw_vi.take(10000)

    vectorizer_en = create_vectorizer(sample_en,VOCAB_SIZE,MAX_LENGTH,VOCAB_EN_FILE)
    vectorizer_vi = create_vectorizer(sample_vi,VOCAB_SIZE,MAX_LENGTH,VOCAB_VI_FILE)

    def vectorize_text(en,vi):
        en = vectorizer_en(en)
        vi = vectorizer_vi(vi)
        return en,vi

    preprocess_dataset = datasets.map(vectorize_text,num_parallel_calls=tf.data.AUTOTUNE)
    preprocess_dataset = preprocess_dataset.shuffle(BUFFER_SIZE)
    preprocess_dataset = preprocess_dataset.batch(BATCH_SIZE)
    preprocess_dataset = preprocess_dataset.prefetch(tf.data.AUTOTUNE)

    return preprocess_dataset, vectorizer_en, vectorizer_vi

if __name__ == "__main__":
    ds, en_shape, vi_shape = get_dataset()
    if ds:
        logging.info("Data Pipline ready!")
        for batch_en, batch_vi in ds.take(1):
            logging.info(f"Shape batch English: {batch_en.shape}")
            logging.info(f"Shape batch Vietnamese: {batch_vi.shape}")

            print("English example", " ".join([en_shape.get_vocabulary()[i] for i in batch_en[0].numpy() if i!=0]))