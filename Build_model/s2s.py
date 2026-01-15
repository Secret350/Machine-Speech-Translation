import os
import time
import asyncio
from idlelib.config_key import translate_key
import speech_recognition as sr
import whisper
import torch
import tensorflow as tf
from config import *
from translation_model import Transformer
from dataset import get_vocab_size,create_vectorizer

try:
    from egde_tts_eng import VNTSS
except ImportError:
    print("edge_tts_eng: file not found!")
    exit()

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
ckpt_en_vi = "./Checkpoint/Train"
ckpt_vi_en = "./Checkpoint_Vi-to-En/Train"

whisper_device = "cpu"

class StsSystem:
    def __init__(self):
        print("Speech to Speech 2-way Start!")

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu,True)
            except RuntimeError as e:
                print(f"Error: {e}")

        print("[1/5] Loading Whisper (ASR)...")
        self.asr_model = whisper.load_model("small",device=whisper_device)

        print("[2/5] Loading Edge-TTS (Onl)...")
        self.tts_eng = VNTSS()

        print("[3/5] EN -> VI Loading Translator En-Vi...")
        self.model_en_vi = self.load_translator(
            ckpt_dir=ckpt_en_vi,
            vocab_in = VOCAB_EN_FILE,
            vocab_out = VOCAB_VI_FILE,
            name="EN-VI"
        )

        print("[4/5] VI -> EN Loading Translator Vi-En...")
        if os.path.exists(ckpt_vi_en):
            self.model_vi_en = self.load_translator(
                ckpt_dir=ckpt_vi_en,
                vocab_in=VOCAB_VI_NEW_FILE,
                vocab_out=VOCAB_EN_NEW_FILE,
                name="VI-EN",
            )
        else:
            print("Model Vi-En not ready translate only from En to Vi!")
        print("[5/5] Calibrating Micro...")
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        print("System Start...\nReady: ")
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
        except Exception as e:
            print(f"Load Error: Model {name}; error {e}")
            return None
    def run(self):
        with self.mic as source:
            print("Preparing microphone please keep silent in 2 seconds...")
            self.recognizer.adjust_for_ambient_noise(source,duration=2)
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
        while True:
            try:
                print("Listening...")
                with self.mic as source:
                    audio = self.recognizer.listen(source,timeout=None,phrase_time_limit=5)
                    with open("temp_input.wav","wb") as f:
                        f.write(audio.get_wav_data())
                    audio_w = whisper.load_audio("temp_input.wav")
                    audio_w = whisper.pad_or_trim(audio_w)
                    mel = whisper.log_mel_spectrogram(audio_w).to(self.asr_model.device)

                    _,probs = self.asr_model.detect_language(mel)
                    if isinstance(probs, list):
                        probs = probs[0]
                    lang = max(probs, key = probs.get)

                    result = self.asr_model.transcribe("temp_input.wav",fp16=False)
                    input_text = result["text"].strip()

                    if not input_text: continue

                    if lang == "en":
                        print(f"EN Input: {input_text}")
                        if self.model_en_vi:
                            translate_text = self.translate_greedy(input_text,self.model_en_vi)
                            print(f"VI output: {translate_text}")
                            self.play_audio_result(translate_text,lang_code="vi")
                        else:
                            print("Model EN-VI not ready now!")
                    elif lang == "vi":
                        print(f"VI Input: {input_text}")
                        if self.model_vi_en:
                            en_text = self.translate_greedy(input_text, self.model_vi_en)
                            print(f"EN Output: {en_text}")
                            self.play_audio_result(en_text, lang_code="en")
                        else:
                            print("Model VI-EN not ready!")
                    else:
                        print(f"Language detected: {lang} is not support now!")
            except KeyboardInterrupt:
                print("Goodbye!")
                break
            except Exception as e:
                print(f"Runtime Error: {e}")
                with self.mic as source: self.recognizer.adjust_for_ambient_noise(source)
    @staticmethod
    def translate_greedy(sentence, system):
        sentence = sentence.lower().strip().replace("?","").replace(".","")
        original_sentence = sentence.strip()
        sentence = original_sentence.lower().strip()

        prefix_translation = ""

        GREETING_HACKS = {
            "what's your name": "bạn tên là gì",
            "what is your name": "bạn tên là gì",
            "hello": "xin chào",
            "hi": "chào bạn",
            "thank you": "cảm ơn",
            "what's up": "xin chào bro"
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
        return final_result.strip()
    def play_audio_result(self,text,lang_code):
        if not text: return
        output_filename = f"out_{lang_code}.mp3"

        saved_file = self.tts_eng.speak(text,output_filename,lang=lang_code)
        if saved_file and os.path.exists(saved_file):
            os.system(f'start /min "" "{saved_file}"')
        else:
            print("Can't create sound!")

if __name__ == "__main__":
    app = StsSystem()
    app.run()