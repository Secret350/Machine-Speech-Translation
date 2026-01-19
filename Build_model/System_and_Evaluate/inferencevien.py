import traceback
import tensorflow as tf
from  Build_model.config import *
from Build_model.Model_path.translation_model import Transformer
from Build_model.Data_Process.dataset import create_vectorizer
import logging
import os
from tokenizers import Tokenizer
import re
import math

HARD_RULES = {
    "xin chào":"hello",
    "chào":"hi",
    "chào buổi sáng":"good morning",
    "chúc ngủ ngon":"good night",
    "cảm ơn bạn":"thank you",
    "cảm ơn":"thanks",
    "tạm biệt":"bye",
    "tạm biệt nhé":"goodbye",
    "tôi yêu bạn":"i love you"
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
        tokenizer_en = Tokenizer.from_file("../ModelCheckpoints/json/tokenizer_en.json")
        tokenizer_vi = Tokenizer.from_file("../ModelCheckpoints/json/tokenizer_vi.json")
    except Exception:
        print(".json file not found! run preprocess_advanced.py first!")
        return None, None, None, None, None, None

    print("\n>>> [2/4] Loading Keras Vectorizers...")

    vectorizer_en = create_vectorizer(None, VOCAB_SIZE, MAX_LENGTH, VOCAB_EN_NEW_FILE)
    vectorizer_vi = create_vectorizer(None, VOCAB_SIZE, MAX_LENGTH, VOCAB_VI_NEW_FILE)

    vocab_en = vectorizer_en.get_vocabulary()
    idx_to_word_en = {i: word for i,word in enumerate(vocab_en)}

    print(">>> [3/4] Dang khoi tao mo hinh Transformer...")

    input_vocab_size = len(vectorizer_vi.get_vocabulary())
    target_vocab_size = len(vectorizer_en.get_vocabulary())
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
    checkpoint_path = "../ModelCheckpoints/VI_EN_Checkpoint"

    temp_optimizer = tf.keras.optimizers.Adam()

    ckpt = tf.train.Checkpoint(transformer=transformer,optimizer=temp_optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt,checkpoint_path,max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print(f">>> Da khoi phuc thanh cong tu {ckpt_manager.latest_checkpoint}")
    else:
        print("\n WARNING: Khong tim thay checkpoint nao! Ket qua dich se sai!")
    return tokenizer_en,tokenizer_vi,vectorizer_en,vectorizer_vi,idx_to_word_en,transformer

def translate(sentence , tokenizer_en, tokenizer_vi,vectorizer_en,vectorizer_vi, idx_to_word_en, transformer):
    clean_sentence = clean_text(sentence)
    if clean_sentence in HARD_RULES:
        result = HARD_RULES[clean_sentence]
        print(f"Input: {sentence}")
        print(f"English prediction (Dictionary): {result}")
        print("\n" + "-" * 50)
        return result
    bpe_tokens = tokenizer_vi.encode(clean_sentence).tokens
    bpe_string = "sostoken " + " ".join(bpe_tokens) + " eostoken"

    vi_seq = vectorizer_vi(tf.convert_to_tensor([bpe_string]))

    vocab_list = vectorizer_en.get_vocabulary()
    print(f"DEBUG Input IDs: {vi_seq.numpy()}")
    try:
        sostoken_token_id = vocab_list.index('sostoken')
        eostoken_token_id = vocab_list.index('eostoken')
    except ValueError:
        try:
            sostoken_token_id = vocab_list.index('sostoken')
            eostoken_token_id = vocab_list.index('eostoken')
        except ValueError:
            sostoken_token_id = 0
            eostoken_token_id = 0

    output_array = tf.convert_to_tensor([[sostoken_token_id]],dtype=tf.int64)
    print(f"Input: {sentence}")
    print(f"Input BPE: {bpe_string}")

    result_ids_hf = []

    #Vong lap sinh tu
    for i in range(MAX_LENGTH):
        predictions, attention_weights = transformer(
            inp=vi_seq,
            tar=output_array,
            training=False
        )
        predictions = predictions[:,-1:,:]
        probs = predictions[0,0].numpy()
        predicted_id = tf.argmax(predictions,axis=-1).numpy()[0][0]

        #Dieu kien dung, neu mo hinh du doan ra padding <0> thi dung
        if i>0:
            previous_token_id = output_array[0,-1].numpy()
            if predicted_id == previous_token_id and predicted_id != eostoken_token_id:
                top2_id = probs.argsort()[-2:][::-1]
                predicted_id = top2_id[1]
        if predicted_id == eostoken_token_id:
            break
        new_token_tensor = tf.constant([[predicted_id]], dtype=tf.int64)
        output_array = tf.concat([output_array,new_token_tensor], axis=-1)
        if predicted_id in idx_to_word_en:
            word = idx_to_word_en[predicted_id]
            if word not in ["","[UNK]","sostoken","eostoken"]:
                hf_id = tokenizer_en.token_to_id(word)
                if hf_id is not None:
                    result_ids_hf.append(hf_id)
    final_sentence = tokenizer_en.decode(result_ids_hf)
    return final_sentence

def beam_search(sentence,tokenizer_en,tokenizer_vi, vectorizer_en, vectorizer_vi, idx_to_word_en, transformer, beam_width=3, alpha=0.6):
    clean_sentence = clean_text(sentence)
    if clean_sentence in HARD_RULES:
        return HARD_RULES[clean_sentence]
    bpe_tokens = tokenizer_vi.encode(clean_sentence).tokens
    bpe_string = "sostoken " + " ".join(bpe_tokens) + " eostoken"
    input_tensor = vectorizer_vi(tf.convert_to_tensor([bpe_string]))

    vocab_list = vectorizer_en.get_vocabulary()

    sostoken_token = None
    eostoken_token = None

    if "sostoken" in vocab_list:
        sostoken_token = vocab_list.index("sostoken")
    elif "sostoken" in vocab_list:
        sostoken_token = vocab_list.index("sostoken")

    if "<eostoken>" in vocab_list:
        eostoken_token = vocab_list.index("<eostoken>")
    elif "eostoken" in vocab_list:
        eostoken_token = vocab_list.index("eostoken")

    if sostoken_token is None or eostoken_token is None:
        return "CRITICAL ERROR: Không tìm thấy token sostoken/eostoken trong vocab_en_to_vi."

    sostoken_seq = tf.constant([[sostoken_token]],dtype=tf.int64)
    beam = [(sostoken_seq,0.0)]

    complete_sequence = []

    for i in range(MAX_LENGTH):
        candidates = []
        for seq, score in beam:
            if seq[0,-1] == eostoken_token:
                complete_sequence.append((seq,score))
                continue
            predictions,_ = transformer(
                inp= input_tensor,
                tar= seq,
                training= False,
            )
            predictions = predictions[:,-1:,:]
            logits = predictions[0,0]
            log_probs = tf.nn.log_softmax(logits).numpy()

            top_k_indices = log_probs.argsort()[-beam_width:][::-1]

            for idx in top_k_indices:
                new_score = score+log_probs[idx]
                new_token = tf.constant([[idx]],dtype=tf.int64)
                new_seq = tf.concat([seq,new_token],axis=-1)
                candidates.append((new_seq,new_score))
        if not candidates:
            break
        beam = sorted(candidates,key=lambda x:x[1],reverse=True)[:beam_width]
    if not complete_sequence:
        complete_sequence=beam
    best_seq = None
    best_score = -float("inf")

    for seq, score in complete_sequence:
        length = seq.shape[1]
        penalty = math.pow(length,alpha)
        final_score = score / penalty

        if final_score>best_score:
            best_score=final_score
            best_seq=seq
    result_ids = best_seq.numpy()[0]
    if len(result_ids) >0:
        result_ids = result_ids[1:]
    result_ids_hf = []
    for idx in result_ids:
        if idx == eostoken_token:
            break
        if idx in idx_to_word_en:
            word = idx_to_word_en[idx]
            if word not in ["","[UNK]","sostoken","eostoken"]:
                hf_id = tokenizer_en.token_to_id(word)
                if hf_id is not None:
                    result_ids_hf.append(hf_id)
    final_sentence = tokenizer_en.decode(result_ids_hf)
    return final_sentence
if __name__ == "__main__":
    tok_en, tok_vi, vec_en, vec_vi, idxtoword, model = load_resource()
    print("---VIETNAMESE TO ENGLISH MACHINE TRANSLATION---")
    while True:
        try:
            text = input("Nhap cau tieng Viet (type 'ext' to cancel): ")
            if text.lower() == "ext":
                break

            if not text.strip():
                continue

            print("Greedy Search:")
            greedy = translate(text,tok_en,tok_vi,vec_en,vec_vi,idxtoword,model)
            print(f"{greedy}")

            # beam_src = beam_search(text,tok_en,tok_vi,vec_en,vec_vi,idxtoword,model,beam_width=3)
            # print(f"Beam Search:{beam_src}")
        except KeyboardInterrupt:
            print("Stop program!")
            break
        except Exception as e:
            print(f"\n Translation error: {e}")
            traceback.print_exc()