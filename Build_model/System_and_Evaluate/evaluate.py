import tensorflow as tf
import random
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
from inference import beam_search, load_resource, clean_text,translate
from Build_model.config import *
from tqdm import tqdm
import re
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def remove_tags(text):
    return text.replace("sostoken", "").replace("eostoken", "").strip()


def calculate_bleu():
    print(">>> Đang load Model & Resources...")
    try:
        tokenizer_en, tokenizer_vi, vectorizer_en, vectorizer_vi, idx_to_word_viet, transformer = load_resource()
    except Exception as e:
        print(f"Lỗi: {e}")
        return

    print(f">>> Đang đọc dữ liệu từ {CLEAN_EN_FILE} và {CLEAN_VI_FILE}...")

    with open(CLEAN_EN_FILE, 'r', encoding='utf-8') as f:
        en_sentences = f.readlines()
    with open(CLEAN_VI_FILE, 'r', encoding='utf-8') as f:
        vi_sentences = f.readlines()

    assert len(en_sentences) == len(vi_sentences), "Lỗi: Số dòng file EN và VI không khớp nhau!"

    NUM_SAMPLES = 100
    total_lines = len(en_sentences)
    indices = random.sample(range(total_lines), min(NUM_SAMPLES, total_lines))

    total_bleu = 0
    smooth = SmoothingFunction().method1

    references_corpus = []
    candidates_corpus = []

    print(f"\n>>> Bắt đầu chấm điểm trên {len(indices)} câu ngẫu nhiên...")
    print("-" * 60)
    print(f"{'INPUT (EN)':<30} | {'PREDICTION (VI)':<30} | {'BLEU':<5}")
    print("-" * 60)

    for idx in tqdm(indices):
        raw_src = en_sentences[idx].strip()
        raw_ref = vi_sentences[idx].strip()

        src_clean = remove_tags(raw_src)

        ref_clean = remove_tags(raw_ref)

        pred_text = translate(
            src_clean,
            tokenizer_en, tokenizer_vi,
            vectorizer_en, vectorizer_vi,
            idx_to_word_viet, transformer
        )

        reference = [ref_clean.split()]
        candidate = pred_text.split()

        references_corpus.append(reference)
        candidates_corpus.append(candidate)

        score = sentence_bleu(reference, candidate, smoothing_function=smooth)
        total_bleu += score

        if idx in indices[:5]:
            print(f"{src_clean[:28]:<30} | {pred_text[:28]:<30} | {score:.2f}")

    avg_sentence_bleu = total_bleu / len(indices)

    corp_bleu = corpus_bleu(references_corpus, candidates_corpus, smoothing_function=smooth)

    print("\n" + "=" * 60)
    print(f"KẾT QUẢ ĐÁNH GIÁ (Trên {len(indices)} câu):")
    print(f"1. Average Sentence BLEU: {avg_sentence_bleu * 100:.2f}")
    print(f"2. Corpus BLEU Score:     {corp_bleu * 100:.2f} (Con số này quan trọng nhất)")
    print("=" * 60)

    final_score = corp_bleu * 100

    if final_score > 40:
        print("=> Đánh giá: XUẤT SẮC (Model dịch rất sát nghĩa)")
    elif final_score > 30:
        print("=> Đánh giá: RẤT TỐT (Model dùng được trong thực tế)")
    elif final_score > 20:
        print("=> Đánh giá: KHÁ (Hiểu ý chính, còn sai ngữ pháp)")
    elif final_score > 10:
        print("=> Đánh giá: TRUNG BÌNH (Model bắt đầu học được từ vựng)")
    else:
        print("=> Đánh giá: YẾU (Cần train thêm hoặc xem lại dữ liệu)")


if __name__ == "__main__":
    calculate_bleu()