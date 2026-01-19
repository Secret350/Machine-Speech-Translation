import sys
import os
import random
import asyncio
import shutil
import edge_tts
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu

# --- 1. FIX LỖI THIẾU DLL (Cho Windows & Faster-Whisper) ---
try:
    import site

    site_packages = site.getsitepackages()[1]
    nvidia_libs = [
        os.path.join(site_packages, "nvidia", "cublas", "bin"),
        os.path.join(site_packages, "nvidia", "cudnn", "bin")
    ]
    for lib_dir in nvidia_libs:
        if os.path.exists(lib_dir):
            os.environ["PATH"] = lib_dir + os.pathsep + os.environ["PATH"]
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(lib_dir)
except Exception:
    pass

# --- 2. CẤU HÌNH IMPORT ---
ROOT_DIR = os.getcwd()
BUILD_MODEL_DIR = ROOT_DIR
sys.path.append(BUILD_MODEL_DIR)

# Chuyển vào Build_model để load code
os.chdir(BUILD_MODEL_DIR)
try:
    from s2s import StsSystem
    import config
except ImportError as e:
    print(f"LỖI: Không tìm thấy file s2s.py hoặc config.py ({e})")
    sys.exit()
os.chdir(ROOT_DIR)

# --- 3. CẤU HÌNH TEST ---
NUM_SAMPLES = 50  # Số câu muốn test
TEMP_AUDIO_DIR = "temp_eval_audio"
LOG_FILE = "evaluation_report.txt"  # Tên file lưu kết quả


def clean_text(text):
    text = text.lower().replace("sostoken", "").replace("eostoken", "").strip()
    import re
    text = re.sub(r"([?.!,])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    return text


async def generate_audio(text, filepath):
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    await communicate.save(filepath)


def log_print(f, text):
    """Hàm phụ trợ: Vừa in ra màn hình, vừa ghi vào file"""
    print(text)
    f.write(text + "\n")


async def run_evaluation():
    # Mở file để ghi log
    log_f = open(LOG_FILE, "w", encoding="utf-8")

    log_print(log_f, "\n" + "=" * 70)
    log_print(log_f, "🚀 BẮT ĐẦU ĐÁNH GIÁ CHI TIẾT TỪNG CÂU (S2S)")
    log_print(log_f, "=" * 70)

    # A. Load Model
    os.chdir(BUILD_MODEL_DIR)
    try:
        system = StsSystem()
    except Exception as e1:
        print(f"Lỗi khởi tạo: {e1}")
        return
    os.chdir(ROOT_DIR)

    # B. Load Data
    log_print(log_f, "\n>>> Đang đọc dữ liệu train...")
    en_path = os.path.join(BUILD_MODEL_DIR, config.CLEAN_EN_FILE) if not os.path.isabs(
        config.CLEAN_EN_FILE) else config.CLEAN_EN_FILE
    vi_path = os.path.join(BUILD_MODEL_DIR, config.CLEAN_VI_FILE) if not os.path.isabs(
        config.CLEAN_VI_FILE) else config.CLEAN_VI_FILE

    with open(en_path, 'r', encoding='utf-8') as f:
        en_lines = f.readlines()
    with open(vi_path, 'r', encoding='utf-8') as f:
        vi_lines = f.readlines()

    indices = random.sample(range(len(en_lines)), min(NUM_SAMPLES, len(en_lines)))

    if os.path.exists(TEMP_AUDIO_DIR): shutil.rmtree(TEMP_AUDIO_DIR)
    os.makedirs(TEMP_AUDIO_DIR)

    refs = []
    cands = []
    smooth = SmoothingFunction().method1

    log_print(log_f, f"\n>>> Đang chạy test trên {len(indices)} câu...")
    log_print(log_f, "-" * 70)

    # --- VÒNG LẶP TEST ---
    for i, idx in enumerate(indices):
        raw_en = en_lines[idx]
        raw_vi = vi_lines[idx]

        src_clean = clean_text(raw_en)
        ref_clean = clean_text(raw_vi)

        if not src_clean: continue

        # 1. Tạo Audio
        audio_path = os.path.join(TEMP_AUDIO_DIR, f"test_{i}.wav")
        await generate_audio(src_clean, audio_path)

        # 2. Whisper Nghe
        try:
            segments, info = system.asr_model.transcribe(audio_path, beam_size=5, vad_filter=True, language="en")
            whisper_text = " ".join([s.text for s in segments]).strip()
        except Exception:
            whisper_text = ""

        # 3. Transformer Dịch
        if whisper_text:
            pred_vi = system.beam_src = system.translate_greedy(whisper_text,system.model_en_vi)
        else:
            pred_vi = ""

        # 4. Tính điểm BLEU cho câu này
        ref_tokens = [ref_clean.split()]
        cand_tokens = pred_vi.split()

        sentence_score = sentence_bleu(ref_tokens, cand_tokens, smoothing_function=smooth)

        refs.append(ref_tokens)
        cands.append(cand_tokens)

        # --- IN KẾT QUẢ CHI TIẾT ---
        log_print(log_f, f"[Câu {i + 1}/{NUM_SAMPLES}]")
        log_print(log_f, f"🎧 Gốc (EN):     {src_clean}")
        log_print(log_f, f"👂 Whisper nghe: {whisper_text}")
        log_print(log_f, f"🤖 Model dịch:   {pred_vi}")
        log_print(log_f, f"🎯 Đích (Ref):   {ref_clean}")
        log_print(log_f, f"⭐️ BLEU:         {sentence_score:.4f}")
        log_print(log_f, "-" * 50)

    # C. Tổng kết
    corpus_score = corpus_bleu(refs, cands, smoothing_function=smooth) * 100

    log_print(log_f, "\n" + "=" * 70)
    log_print(log_f, f"🏆 TỔNG KẾT TOÀN BỘ (CORPUS BLEU): {corpus_score:.2f} / 100")
    log_print(log_f, "=" * 70)

    log_print(log_f, f"\n(Đã lưu chi tiết kết quả vào file '{LOG_FILE}')")

    # Đóng file và dọn dẹp
    log_f.close()
    if os.path.exists(TEMP_AUDIO_DIR):
        try:
            shutil.rmtree(TEMP_AUDIO_DIR)
        except:
            pass


if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_evaluation())