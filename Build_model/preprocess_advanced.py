import os
import re
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from config import *
from tqdm import tqdm
import logging

VOCAB_SIZE_BPE = 32000
MIN_LEN = 2
MAX_LENGTH_FILTER = 100
LEN_RATIO = 1.5
MAX_SAMPLE = 2000000

logging.basicConfig(level=logging.INFO)

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r"([?.!,;])", r" \1 ", text)
    text = re.sub(r'["\n\r]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def filter_pair(en,vi):
    len_en = len(en.split())
    len_vi = len(vi.split())

    if (len_en < MIN_LEN) or (len_vi < MIN_LEN):
        return False
    if (len_en > MAX_LENGTH_FILTER) or (len_vi > MAX_LENGTH_FILTER):
        return False

    if (len_vi/len_en > LEN_RATIO) or (len_en / len_vi > LEN_RATIO):
        return False

    return True

def train_and_save_bpe(files,save_path,vocab_size):
    logging.info(f"Training BPE Tokenizer for {files}")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]","[PAD]","<start>","<end>"]
    )

    tokenizer.train(files, trainer)

    tokenizer.save(save_path)
    logging.info(f"BPE was saved in {save_path}")
    return tokenizer

def process_pipline():
    logging.info("Cleaning and Filtering...")

    with open(RAW_EN_FILE,"r", encoding="utf-8", errors="ignore") as f_en,\
        open(RAW_VI_FILE,"r", encoding="utf-8", errors="ignore") as f_vi,\
        open("temp_clean.en","w",encoding="utf-8") as t_en,\
        open("temp_clean.vi","w",encoding="utf-8") as t_vi:

        lines_en = f_en.readlines()
        lines_vi = f_vi.readlines()

        kept_count = 0

        for en, vi in tqdm(zip(lines_en,lines_vi,),desc="Filtering"):
            clean_en = clean_text(en)
            clean_vi = clean_text(vi)

            if filter_pair(clean_en,clean_vi):
                t_en.write(clean_en+"\n")
                t_vi.write(clean_vi+"\n")
                kept_count+=1
            if kept_count >= MAX_SAMPLE:
                logging.info(f"Reach limit {MAX_SAMPLE} samples!")
                break
    logging.info(f"Kept {kept_count} clean sentences / {len(lines_en)} sentences.")
    logging.info("Training BPE Tokenizer...")
    tokenizer_en = train_and_save_bpe(["temp_clean.en"],"tokenizer_en.json",VOCAB_SIZE_BPE)
    tokenizer_vi = train_and_save_bpe(["temp_clean.vi"],"tokenizer_vi.json",VOCAB_SIZE_BPE)

    logging.info("Encoding BPE and add <start>, <end> tags...")

    with open("temp_clean.en", "r", encoding="utf-8") as f_en, \
        open("temp_clean.vi", "r", encoding="utf-8") as f_vi, \
        open(CLEAN_EN_FILE, "w", encoding="utf-8") as out_en, \
        open(CLEAN_VI_FILE, "w", encoding="utf-8") as out_vi:

        lines_en = f_en.readlines()
        lines_vi = f_vi.readlines()

        for en, vi in tqdm(zip(lines_en,lines_vi),total=len(lines_en)):
            en = en.strip()
            vi = vi.strip()

            tokens_en = tokenizer_en.encode(en).tokens
            tokens_vi = tokenizer_vi.encode(vi).tokens

            clean_bpe_en = " ".join(tokens_en)
            clean_bpe_vi = " ".join(tokens_vi)

            out_en.write(clean_bpe_en+"\n")
            out_vi.write(f"<start> {clean_bpe_vi} <end>\n")
    if os.path.exists("temp_clean.en"):
        os.remove("temp_clean.en")
    if os.path.exists("temp_clean.vi"):
        os.remove("temp_clean.vi")
    logging.info(f"Completed! Data ready at {CLEAN_EN_FILE} and {CLEAN_VI_FILE}")

if __name__ == "__main__":
    process_pipline()
