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
# Nếu bạn chạy từ root thì có thể cần sửa lại: os.path.join(ROOT_DIR, 'Build_model')
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
NUM_SAMPLES = 50  # Số câu muốn test cho mỗi chiều
TEMP_AUDIO_DIR = "temp_eval_audio"
LOG_FILE = "evaluation_report_2way.txt"


def clean_text(text):
    text = text.lower().replace("sostoken", "").replace("eostoken", "").strip()
    import re
    text = re.sub(r"([?.!,])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    return text.strip()


async def generate_audio(text, filepath, lang_code):
    """
    Tạo audio dựa trên ngôn ngữ đầu vào.
    - lang_code = 'en' -> Dùng giọng Mỹ (Aria)
    - lang_code = 'vi' -> Dùng giọng Việt (Hoài My)
    """
    if lang_code == "en":
        voice = "en-US-AriaNeural"
    else:
        # Giọng nữ miền Nam chuẩn, dễ nghe cho Whisper
        voice = "vi-VN-HoaiMyNeural"

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(filepath)


def log_print(f, text):
    """Hàm phụ trợ: Vừa in ra màn hình, vừa ghi vào file"""
    print(text)
    f.write(text + "\n")


async def evaluate_one_direction(direction, system, indices, en_lines, vi_lines, log_f):
    """Hàm chạy đánh giá cho 1 chiều cụ thể để tránh lặp code"""

    # Cấu hình dựa trên chiều dịch
    if direction == "en_vi":
        src_lines = en_lines
        tgt_lines = vi_lines
        src_lang = "en"  # Audio đầu vào là tiếng Anh
        tgt_lang = "vi"
        # Đảm bảo class StsSystem của bạn có thuộc tính model_en_vi
        model_mt = system.model_en_vi
        title = "CHIỀU: ANH (Audio) -> VIỆT (Text)"
    else:  # vi_en
        src_lines = vi_lines
        tgt_lines = en_lines
        src_lang = "vi"  # Audio đầu vào là tiếng Việt
        tgt_lang = "en"
        # Đảm bảo class StsSystem của bạn có thuộc tính model_vi_en
        model_mt = system.model_vi_en
        title = "CHIỀU: VIỆT (Audio) -> ANH (Text)"

    log_print(log_f, "\n" + "=" * 80)
    log_print(log_f, title)
    log_print(log_f, "=" * 80)

    refs = []
    cands = []
    smooth = SmoothingFunction().method1

    # Dọn dẹp folder tạm trước khi chạy chiều mới
    if os.path.exists(TEMP_AUDIO_DIR): shutil.rmtree(TEMP_AUDIO_DIR)
    os.makedirs(TEMP_AUDIO_DIR)

    log_print(log_f, f"{'INPUT (Audio)':<35} | {'WHISPER HEARD':<35} | {'OUTPUT':<35} | {'BLEU'}")
    log_print(log_f, "-" * 120)

    for i, idx in enumerate(indices):
        raw_src = src_lines[idx]
        raw_tgt = tgt_lines[idx]

        src_clean = clean_text(raw_src)
        ref_clean = clean_text(raw_tgt)

        if not src_clean: continue

        # 1. Tạo Audio (Chọn giọng theo src_lang)
        audio_path = os.path.join(TEMP_AUDIO_DIR, f"test_{src_lang}_{i}.wav")
        await generate_audio(src_clean, audio_path, src_lang)

        # 2. Whisper Nghe (Quan trọng: Phải ép ngôn ngữ đúng chiều)
        try:
            segments, info = system.asr_model.transcribe(
                audio_path,
                beam_size=5,
                vad_filter=True,
                language=src_lang  # Ép Whisper nghe đúng ngôn ngữ nguồn
            )
            whisper_text = " ".join([s.text for s in segments]).strip()
        except Exception as e:
            whisper_text = ""

        # 3. Transformer Dịch
        if whisper_text:
            # Dùng đúng model cho chiều tương ứng
            pred = system.translate_greedy(whisper_text, model_mt)
        else:
            pred = ""

        # Chuẩn hóa output để tính điểm
        pred = clean_text(pred)

        # 4. Tính điểm BLEU
        ref_tokens = [ref_clean.split()]
        cand_tokens = pred.split()

        sentence_score = sentence_bleu(ref_tokens, cand_tokens, smoothing_function=smooth)

        refs.append(ref_tokens)
        cands.append(cand_tokens)

        # In mẫu 10 câu đầu tiên ra màn hình để soi
        if i < 10:
            # Cắt ngắn text hiển thị cho đỡ rối mắt
            display_src = src_clean[:30] + "..." if len(src_clean) > 30 else src_clean
            display_hrd = whisper_text[:30] + "..." if len(whisper_text) > 30 else whisper_text
            display_out = pred[:30] + "..." if len(pred) > 30 else pred

            log_print(log_f, f"{display_src:<35} | {display_hrd:<35} | {display_out:<35} | {sentence_score:.4f}")

    # Tổng kết chiều này
    corpus_score = corpus_bleu(refs, cands, smoothing_function=smooth) * 100
    log_print(log_f, "-" * 120)
    log_print(log_f, f"✨ KẾT QUẢ {title}: {corpus_score:.2f} BLEU")

    return corpus_score


async def run_evaluation():
    # Mở file ghi log
    log_f = open(LOG_FILE, "w", encoding="utf-8")
    log_print(log_f, "BẮT ĐẦU ĐÁNH GIÁ HỆ THỐNG S2S 2 CHIỀU (End-to-End)")

    # A. Load Model (Load 1 lần dùng chung cho cả 2 chiều)
    os.chdir(BUILD_MODEL_DIR)
    try:
        system = StsSystem()
    except Exception as e1:
        print(f"Lỗi khởi tạo: {e1}")
        return
    os.chdir(ROOT_DIR)

    # B. Load Data
    log_print(log_f, "\n>>> Đang đọc dữ liệu train từ config...")
    # Xử lý đường dẫn tương đối/tuyệt đối
    en_path = os.path.join(BUILD_MODEL_DIR, config.CLEAN_EN_FILE) if not os.path.isabs(
        config.CLEAN_EN_FILE) else config.CLEAN_EN_FILE
    vi_path = os.path.join(BUILD_MODEL_DIR, config.CLEAN_VI_FILE) if not os.path.isabs(
        config.CLEAN_VI_FILE) else config.CLEAN_VI_FILE

    with open(en_path, 'r', encoding='utf-8') as f:
        en_lines = f.readlines()
    with open(vi_path, 'r', encoding='utf-8') as f:
        vi_lines = f.readlines()

    # Chọn mẫu ngẫu nhiên (dùng chung index cho cả 2 chiều để công bằng)
    indices = random.sample(range(len(en_lines)), min(NUM_SAMPLES, len(en_lines)))

    # --- CHẠY TEST CHIỀU 1: EN -> VI ---
    score_en_vi = await evaluate_one_direction("en_vi", system, indices, en_lines, vi_lines, log_f)

    # --- CHẠY TEST CHIỀU 2: VI -> EN ---
    score_vi_en = await evaluate_one_direction("vi_en", system, indices, en_lines, vi_lines, log_f)

    # --- TỔNG KẾT CHUNG ---
    log_print(log_f, "\n" + "=" * 70)
    log_print(log_f, "BẢNG TỔNG SẮP END-TO-END (S2S)")
    log_print(log_f, "=" * 70)
    log_print(log_f, f"1. Anh -> Việt: {score_en_vi:.2f} BLEU")
    log_print(log_f, f"2. Việt -> Anh: {score_vi_en:.2f} BLEU")
    log_print(log_f, "-" * 40)
    log_print(log_f, f"TRUNG BÌNH:   {(score_en_vi + score_vi_en) / 2:.2f} BLEU")
    log_print(log_f, "=" * 70)

    log_print(log_f, f"\n(Chi tiết xem tại file '{LOG_FILE}')")
    log_f.close()

    # Dọn dẹp folder audio tạm
    if os.path.exists(TEMP_AUDIO_DIR):
        try:
            shutil.rmtree(TEMP_AUDIO_DIR)
        except:
            pass


if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_evaluation())