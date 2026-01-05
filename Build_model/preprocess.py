from config import *
from underthesea import word_tokenize
from tqdm import tqdm
import logging
import unicodedata

def is_valid_utf8(text):
    try:
        text.encode('utf-8')
        return True
    except UnicodeError:
        return False

def preprocess(raw_en_path, raw_vi_path, out_en_path, out_vi_path):
    logging.info("Preprocessing...!")
    with open(raw_en_path,"r", encoding="utf-8",errors="replace") as f_en,\
        open(raw_vi_path,"r",encoding="utf-8",errors="replace") as f_vi,\
        open(out_en_path,"w",encoding="utf-8") as out_en,\
        open(out_vi_path,"w",encoding="utf-8") as out_vi:

        lines_en = f_en.readlines()
        lines_vi = f_vi.readlines()

        assert len(lines_en) == len(lines_vi), "The number of lines in two file is not fit!"

        count = 0
        skipped_errors = 0
        for en,vi in tqdm(zip(lines_en,lines_vi),total=len(lines_en)):
            en = en.strip()
            vi = vi.strip()
            vi = unicodedata.normalize('NFC', vi)
            en = unicodedata.normalize('NFC', en)

            if not en or not vi or len(en.split()) > MAX_LENGTH:
                continue

            if '\ufffd' in en or '\ufffd' in vi:
                skipped_errors += 1
                continue

            try:
                vi_segmented = word_tokenize(vi, format="text")
                vi_segmented.encode('utf-8')
                en.encode('utf-8')

                out_en.write(en + "\n")
                out_vi.write(vi_segmented + "\n")
                count += 1
            except Exception:
                skipped_errors += 1
                continue

            if count >= 5000:
                break
    logging.info(f"Preprocessing success! Number of sentences: {count}")
    logging.info(f"Save at {PROCESSED_DIR}")

if __name__ == "__main__":
    preprocess(RAW_EN_FILE,RAW_VI_FILE,CLEAN_EN_FILE,CLEAN_VI_FILE)