import csv
import os
import math
import re

from config import *
from termcolor import colored as color
from prompt_toolkit import prompt
from datetime import datetime
try:
    import site
    site_packages = site.getsitepackages()[1]

    nvidia_libs = [
        os.path.join(site_packages,"nvidia","cublas","bin"),
        os.path.join(site_packages,"nvidia","cudnn","bin")
    ]

    for lib_dir in nvidia_libs:
        if os.path.exists(lib_dir):
            os.environ["PATH"] = lib_dir + os.pathsep + os.environ["PATH"]
            if hasattr(os,"add_dll_directory"):
                os.add_dll_directory(lib_dir)
except Exception as e:
    print(f"WARN: Can not load NVIDIA libs: {e}")

import time
import speech_recognition as sr
from faster_whisper import WhisperModel
import tensorflow as tf
from Build_model.Model_path.translation_model import Transformer
from Build_model.Data_Process.dataset import create_vectorizer

Vocab_En = "./ModelCheckpoints/Vocab/vocab_en.pkl"
Vocab_Vi = "./ModelCheckpoints/Vocab/vocab_vi.pkl"

Vocab_En_New = "./ModelCheckpoints/Vocab/vocab_new_en.pkl"
Vocab_Vi_New = "./ModelCheckpoints/Vocab/vocab_new_vi.pkl"

os.environ["HF_TOKEN"] = "---!!!Add your HF_TOKEN!!!---"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
try:
    from Build_model.Model_path.egde_tts_eng import VNTSS
except ImportError:
    print("edge_tts_eng: file not found!")
    exit()

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
ckpt_en_vi = "./ModelCheckpoints/EN_VI_Checkpoint"
ckpt_vi_en = "./ModelCheckpoints/VI_EN_Checkpoint"

whisper_device = "cuda"

class StsSystem:
    def __init__(self):
        print("Speech to Speech 2-way Start!")

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu,True)
            except RuntimeError as error:
                print(f"Error: {error}")

        print("[1/5] Loading Faster Whisper (ASR)...")
        try:
            self.asr_model = WhisperModel("large-v3", device=whisper_device, compute_type="int8")
        except Exception:
            print("GPU Error or not supported, switching to float32...")
            self.asr_model = WhisperModel("large-v3", device=whisper_device, compute_type="float32")

        print("[2/5] Loading Edge-TTS (Onl)...")
        self.tts_eng = VNTSS()

        print("[3/5] EN -> VI Loading Translator En-Vi...")
        self.model_en_vi = self.load_translator(
            ckpt_dir=ckpt_en_vi,
            vocab_in = Vocab_En,
            vocab_out = Vocab_Vi,
            name="EN-VI"
        )

        print("[4/5] VI -> EN Loading Translator Vi-En...")
        if os.path.exists(ckpt_vi_en):
            self.model_vi_en = self.load_translator(
                ckpt_dir=ckpt_vi_en,
                vocab_in=Vocab_Vi_New,
                vocab_out=Vocab_En_New,
                name="VI-EN",
            )
        else:
            print("Model Vi-En not ready translate only from En to Vi!")
        print("[5/5] Calibrating Micro...")
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        print("System Started!")
    @staticmethod
    def load_translator(ckpt_dir, vocab_in, vocab_out, name):
        try:
            dummy = tf.data.Dataset.from_tensor_slices(["dummy"])
            vec_in = create_vectorizer(dummy,VOCAB_SIZE,MAX_LENGTH,vocab_in)
            vec_out = create_vectorizer(dummy,VOCAB_SIZE,MAX_LENGTH,vocab_out)

            vocab_out_list = vec_out.get_vocabulary()
            idx_to_word = {i:w for i,w in enumerate(vocab_out_list)}
            in_size = len(vec_in.get_vocabulary())
            out_size =len(vec_out.get_vocabulary())
            transformer = Transformer(
                num_layers=NUM_LAYERS,
                d_model=D_MODEL,
                num_heads=NUM_HEADS,
                dff=DFF,
                input_vocab_size=in_size,
                target_vocab_size=out_size,
                dropout_rate=DROPOUT_RATE
            )
            dummy_input = tf.zeros((1, MAX_LENGTH), dtype=tf.int64)
            dummy_target = tf.zeros((1, MAX_LENGTH), dtype=tf.int64)
            _, _ = transformer(dummy_input, dummy_target, training=False)
            ckpt = tf.train.Checkpoint(transformer=transformer)
            manager = tf.train.CheckpointManager(ckpt,ckpt_dir,max_to_keep=5)

            if manager.latest_checkpoint:
                ckpt.restore(manager.latest_checkpoint).expect_partial()
                return {"model":transformer, "vec_in":vec_in, "idx_to_word":idx_to_word}
            else:
                print(f"Checkpoint not found!")
                return None
        except Exception as e2:
            print(f"Load Error: Model {name}; error {e2}")
            return None
    @staticmethod
    def check_input(input_sentence):
        print(f"Is the sentence correct? Press {color('[Enter]','green')} to skip / Use arrow button to move and fix then press {color('[Enter]','green')}! ")
        sentence = prompt("You can edit sentence here: ", default=input_sentence)
        return sentence
    @staticmethod
    def save_to_train(source, target, lang_pair):
        filename = f"user_check_dataset_{lang_pair}.csv"
        file_exists = os.path.isfile(filename)
        with open(filename, mode="a",newline='',encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Source","Target"])
            writer.writerow([source,target])
        print(f"Edit saved to train later! {filename}")

    @staticmethod
    def detect_language_by_text(text):
        vietnamese_chars = "áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ"
        text_lower = text.lower()
        for char in vietnamese_chars:
            if char in text_lower:
                return "vi"
        return "en"
    def run(self):
        with self.mic as source:
            print("Preparing microphone please keep silent in 2 seconds...")
            self.recognizer.adjust_for_ambient_noise(source,duration=2)
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 1
        while True:
            try:
                print("Listening...")
                with self.mic as source:
                    audio = self.recognizer.listen(source,timeout=None,phrase_time_limit=10)
                    start_time = time.time()
                    with open("temp_input.wav","wb") as f:
                        f.write(audio.get_wav_data())
                    segments,info = self.asr_model.transcribe(
                        "temp_input.wav",
                        beam_size=5,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500),
                        initial_prompt="Đây là cuộc hội thoại giao tiếp bằng Tiếng Việt và Tiếng Anh. This is a conversation in Vietnamese and English.")
                    text_seg = []
                    for seg in segments:
                        text_seg.append(seg.text)
                    unchecked_text = ' '.join(text_seg).strip()

                    lang = info.language
                    final_lang = self.detect_language_by_text(unchecked_text)
                    t_asr = time.time()
                    lang = final_lang
                    input_text = self.check_input(unchecked_text)

                    if not input_text:
                        continue
                    t_tts = t_asr
                    processed = False

                    if lang == "en":
                        print(f"Input: {input_text}")
                        if self.model_en_vi:
                            vi_text = self.translate_greedy(input_text,self.model_en_vi)
                            # vi_text = self.beam_search(input_text, self.model_en_vi)
                            print(f"Output: {vi_text}")
                            self.play_audio_result(vi_text,lang_code="vi")
                            t_tts = time.time()
                            final_target= self.check_input(vi_text)
                            self.save_to_train(input_text,final_target,"en_vi")
                            processed = True
                        else:
                            print("Model EN-VI not ready now!")
                    elif lang == "vi":
                        print(f"Input: {input_text}")
                        if self.model_vi_en:
                            en_text = self.translate_greedy(input_text, self.model_vi_en)
                            # en_text = self.beam_search(input_text, self.model_vi_en)
                            print(f"Output: {en_text}")
                            self.play_audio_result(en_text, lang_code="en")
                            t_tts = time.time()
                            final_target= self.check_input(en_text)
                            self.save_to_train(input_text, final_target, "vi_en")
                            processed = True
                        else:
                            print("Model VI-EN not ready!")
                    else:
                        print(f"Language detected: {lang} is not support now!")
                    if processed:
                        print(color(f"=> Total:    {t_tts - start_time:.2f}s","cyan",attrs=['bold']))
            except KeyboardInterrupt:
                print("Goodbye!")
                break
            except Exception as e3:
                print(f"Runtime Error: {e3}")
                with self.mic as source: self.recognizer.adjust_for_ambient_noise(source)
    @staticmethod
    def translate_greedy(sentence, system):
        sentence = sentence.lower().strip()
        sentence = re.sub(r'[.?!,]+$', '', sentence)

        prefix_translation = ""

        GREETING_HACKS = {
            "what's your name": "bạn tên là gì",
            "what is your name": "bạn tên là gì",
            "hello": "xin chào",
            "hi": "chào bạn",
            "thank you": "cảm ơn",
            "what's up": "xin chào bro",
            "i'm good":"Tôi ổn",
            "come on":"thôi nào",
            "bạn tên là gì":"what's your name",
            "xin chào":"hello",
            "chào bạn":"hi",
            "cảm ơn":"thank you",
            "xin chào bro": "what's up",
            "thôi nào":"come on"
        }

        for eng_word, vi_word in GREETING_HACKS.items():
            if sentence.startswith(eng_word + " ") or \
                    sentence.startswith(eng_word + ",") or \
                    sentence == eng_word:

                prefix_translation = vi_word

                sentence = sentence[len(eng_word):].strip()

                if sentence.startswith(","):
                    sentence = sentence[1:].strip()
                    prefix_translation += ","
                prefix_translation += " "
                break

        if not sentence:
            return prefix_translation.strip()

        vec_in = system["vec_in"]
        transformer = system["model"]
        idx_to_word = system["idx_to_word"]
        sentence = f"sostoken {sentence} eostoken"

        inp_tensor = vec_in(tf.constant([sentence]))

        start_token = 2
        end_token = 3

        output_array = tf.constant([[start_token]],dtype=tf.int64)
        result_word = []

        for i in range(MAX_LENGTH):
            predictions,_ = transformer(inp_tensor,output_array,training=False)
            predictions = predictions[:,-1:,:]
            predicted_id = tf.argmax(predictions,axis=-1).numpy()[0][0]

            if predicted_id == end_token:
                break
            output_array = tf.concat([output_array,tf.constant([[predicted_id]],dtype=tf.int64)],axis=-1)

            if predicted_id in idx_to_word:
                word = idx_to_word[predicted_id]
                if word not in ["","[UNK]","sostoken","eostoken"]:
                    result_word.append(word)
        ai_trans =  " ".join(result_word)
        final_result = prefix_translation + ai_trans
        final_result = re.sub(r'^[,.\s?]+', '', final_result).strip()
        final_result = final_result.capitalize()
        return final_result.strip()
    @staticmethod
    def beam_search(sentence, system, beam_size=3, alpha=1.2):
        sentence = sentence.lower().strip().replace("?","").replace(".","")
        if not sentence: return ""

        vec_in = system["vec_in"]
        transformer = system["model"]
        idx_to_word = system["idx_to_word"]

        sentence = f"sostoken {sentence} eostoken"
        inp_tensor = vec_in(tf.constant([sentence]))

        start_token = 2
        end_token = 3

        beam = [([start_token],0.0)]
        completed_sequences = []
        for _ in range (MAX_LENGTH):
            candidates = []
            for seq,score in beam:
                if seq[-1] == end_token:
                    norm_score = score/(len(seq)**alpha)
                    completed_sequences.append((seq,norm_score))
                    continue
                tar_tensor = tf.constant([seq],dtype=tf.int64)
                predictions,_ = transformer(inp_tensor,tar_tensor,training=False)
                predictions = predictions[:,-1:,:]
                probs = tf.nn.softmax(predictions,axis=-1).numpy()[0][0]

                top_k_indices = probs.argsort()[-beam_size:][::-1]
                for idx in top_k_indices:
                    prob = probs[idx]
                    log_prob = math.log(prob + 1e-10)
                    new_score = score + log_prob
                    new_seq = seq + [idx]
                    candidates.append((new_seq,new_score))
            if len(candidates) == 0:
                break
            ordered = sorted(candidates,key=lambda x: x[1],reverse=True)
            beam = ordered[:beam_size]
        if not completed_sequences:
            completed_sequences = [(seq,score/(len(seq)**alpha)) for seq, score in beam]
        best_seq, best_score = max(completed_sequences, key=lambda x: x[1])

        result_words = []
        for idx in best_seq:
            if idx in idx_to_word:
                word = idx_to_word[idx]
                if word not in ["sostoken","eostoken","[UNK]"]:
                    result_words.append(word)
        return " ".join(result_words)
    def play_audio_result(self,text,lang_code):
        if not text: return
        output_filename = f"out_{lang_code}.mp3"
        try:
            saved_file = self.tts_eng.speak(text,output_filename,lang=lang_code)
            if saved_file and os.path.exists(saved_file):
                os.system(f'start /min "" "{saved_file}"')
            else:
                print("Can't create sound!")
        except Exception as e4:
            print(f"TTS Error: {e4}")

if __name__ == "__main__":
    app = StsSystem()
    app.run()