import traceback
import tensorflow as tf
from  config import *
from translation_model import Transformer
from dataset import create_vectorizer
import logging
import os
from tokenizers import Tokenizer
import re

HARD_RULES = {
    "hello": "xin chào",
    "hi": "chào bạn",
    "good morning": "chào buổi sáng",
    "good night": "chúc ngủ ngon",
    "thank you": "cảm ơn bạn",
    "thanks": "cảm ơn",
    "bye": "tạm biệt",
    "goodbye": "tạm biệt nhé",
    "i love you": "tôi yêu bạn"
}

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
        print(f"System is using GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(e)
else:
    print("System is using CPU")

def clean_text(txt):
    txt = txt.lower().strip()
    txt = re.sub(r"([?.!,;])", r" \1 ", txt)
    txt = re.sub(r'["\n\r]+', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt)
    return txt

def load_resource():
    print("\n>>>[1/4] Loading BPE Tokenizers...")
    try:
        tokenizer_en = Tokenizer.from_file("tokenizer_en.json")
        tokenizer_vi = Tokenizer.from_file("tokenizer_vi.json")
    except Exception:
        print(".json file not found! run preprocess_advanced.py first!")
        return None, None, None, None, None, None

    print("\n>>> [2/4] Loading Keras Vectorizers...")

    vectorizer_en = create_vectorizer(None, VOCAB_SIZE, MAX_LENGTH, VOCAB_EN_FILE)
    vectorizer_vi = create_vectorizer(None, VOCAB_SIZE, MAX_LENGTH, VOCAB_VI_FILE)

    vocab_vi = vectorizer_vi.get_vocabulary()
    idx_to_word_viet = {i: word for i,word in enumerate(vocab_vi)}

    print(">>> [3/4] Dang khoi tao mo hinh Transformer...")

    input_vocab_size = len(vectorizer_en.get_vocabulary())
    target_vocab_size = len(vectorizer_vi.get_vocabulary())
    print(f"Input Vocab Size: {input_vocab_size}")
    print(f"Target Vocab Size: {target_vocab_size}")

    transformer = Transformer(
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dff=DFF,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        dropout_rate=DROPOUT_RATE
    )
    print(">>> [4/4] Dang load Weights tu Checkpoint...")
    checkpoint_path = "./Checkpoint/Train"

    temp_optimizer = tf.keras.optimizers.Adam()

    ckpt = tf.train.Checkpoint(transformer=transformer,optimizer=temp_optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt,checkpoint_path,max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print(f">>> Da khoi phuc thanh cong tu {ckpt_manager.latest_checkpoint}")
    else:
        print("\n WARNING: Khong tim thay checkpoint nao! Ket qua dich se sai!")
    return tokenizer_en,tokenizer_vi,vectorizer_en,vectorizer_vi,idx_to_word_viet,transformer

def translate(sentence , tokenizer_en, tokenizer_vi,vectorizer_en,vectorizer_vi, idx_to_word_viet, transformer):
    clean_sentence = clean_text(sentence)
    if clean_sentence in HARD_RULES:
        result = HARD_RULES[clean_sentence]
        print(f"Input: {sentence}")
        print(f"Vietnamese prediction (Dictionary): {result}")  # Báo là dùng từ điển
        print("\n" + "-" * 50)
        return result
    bpe_tokens = tokenizer_en.encode(clean_sentence).tokens
    bpe_string = " ".join(bpe_tokens)

    en_seq = vectorizer_en(tf.convert_to_tensor([bpe_string]))

    vocab_list = vectorizer_vi.get_vocabulary()
    print(f"DEBUG Input IDs: {en_seq.numpy()}")
    try:
        start_token_id = vocab_list.index('<start>')
        end_token_id = vocab_list.index('<end>')
    except ValueError:
        try:
            start_token_id = vocab_list.index('start')
            end_token_id = vocab_list.index('end')
        except ValueError:
            start_token_id = 0
            end_token_id = 0

    output_array = tf.convert_to_tensor([[start_token_id]],dtype=tf.int64)
    print(f"Input: {sentence}")
    print(f"Input BPE: {bpe_string}")
    print("Vietnamese prediction: ", end="",flush=True)

    result_ids_hf = []

    #Vong lap sinh tu
    for i in range(MAX_LENGTH):
        predictions, attention_weights = transformer(
            inp=en_seq,
            tar=output_array,
            training=False
        )
        predictions = predictions[:,-1:,:]
        probs = predictions[0,0].numpy()
        predicted_id = tf.argmax(predictions,axis=-1).numpy()[0][0]

        #Dieu kien dung, neu mo hinh du doan ra padding <0> thi dung
        if i>0:
            previous_token_id = output_array[0,-1].numpy()
            if predicted_id == previous_token_id and predicted_id != end_token_id:
                top2_id = probs.argsort()[-2:][::-1]
                predicted_id = top2_id[1]
        if predicted_id == end_token_id:
            break
        new_token_tensor = tf.constant([[predicted_id]], dtype=tf.int64)
        output_array = tf.concat([output_array,new_token_tensor], axis=-1)
        if predicted_id in idx_to_word_viet:
            word = idx_to_word_viet[predicted_id]
            if word not in ["","[UNK]","<start>","<end>"]:
                hf_id = tokenizer_vi.token_to_id(word)
                if hf_id is not None:
                    result_ids_hf.append(hf_id)
    final_sentence = tokenizer_vi.decode(result_ids_hf)
    print(f"=> {final_sentence}")
    print("\n" + "-"*50)
    return final_sentence

if __name__ == "__main__":
    tok_en, tok_vi, vec_en, vec_vi, idxtoword, model = load_resource()
    print("---ENGLISH TO VIETNAMESE MACHINE TRANSLATION---")
    while True:
        try:
            text = input("Nhap cau tieng Anh (type 'ext' to cancel): ")
            if text.lower() == "ext":
                break

            if not text.strip():
                continue

            translate(text,tok_en,tok_vi,vec_en,vec_vi,idxtoword,model)

        except KeyboardInterrupt:
            print("Stop program!")
            break
        except Exception as e:
            print(f"\n Translation error: {e}")
            traceback.print_exc()