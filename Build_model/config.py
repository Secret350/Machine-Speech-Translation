VOCAB_DIR = "./Training_Data/vocab"
VOCAB_EN_FILE = VOCAB_DIR+"vocab_en.pkl"
VOCAB_VI_FILE = VOCAB_DIR+"vocab_vi.pkl"

# 1. Đường dẫn dữ liệu
RAW_DIR = "./Training_Data/raw"
PROCESSED_DIR = "./Training_Data/processed"

# 2 file dataset đã giải nén
RAW_EN_FILE = RAW_DIR+"/OpenSubtitles.en-vi.en"
RAW_VI_FILE = RAW_DIR+"/OpenSubtitles.en-vi.vi"

# 3. File sau khi làm sạch
CLEAN_EN_FILE = PROCESSED_DIR+"/train.en"
CLEAN_VI_FILE = PROCESSED_DIR+"/train.vi"

# 4. Hyperparameters cho Model & Data
VOCAB_SIZE = 40000
MAX_LENGTH = 60
BATCH_SIZE = 128
BUFFER_SIZE = 20000

# 5. Token
START_TOKEN = "<start>"
END_TOKEN = "<end>"

#HYPER PARAMETER
NUM_LAYERS = 4
D_MODEL = 256
DFF = 1024
NUM_HEADS = 8
DROPOUT_RATE = 0.1