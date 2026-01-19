**Machine-Speech-Translation End-to-End (EN -> VI | VI -> EN)**
___
A Speech to Speech and Text to Text Machine Translation system using **FasterWhisper** for ASR, custom **Transformer** model for translation, and **edge-tts** to generate output speech.
___
--- Feature ---
- **Automatic Speech Recognition**: FasterWhisper (Large-v3 int8)
- **Translation**: Custom trained Transformer model (English <-> Vietnamese)
- **Text to Speech**: Edge-TTS for natural output sound
- **Latency**: Optimize for acceptable latency (2-4s for 1 sentence)
___
--- Installation --- 
1. Clone the repository:
    ```bash
    git clone [https://github.com/Secret350/Machine-Speech-Translation.git](https://github.com/Secret350/Machine-Speech-Translation.git)
    cd Machine Speech Translation
    ```
2. Install dependencies:
    ```bash
   pip install -r requirements.txt
   ```
3. Install NVIDIA Libraries (for GPU support):
    ```bash
   pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
   ``` 
___
--- How to run program ---
* Use pre-trained model:

  - **First**: Install weight of pre-trained model "ModelCheckpoints" by the link below then extract and place that directory in Build_model 
  ```
  https://drive.google.com/file/d/1D0G29vXtSGe2wyjwIKx8lJDUWhNtzsKT/view?usp=drive_link
  ```
  - To run Text-to-Text (EN > VI) program
  ```bash
  cd Build_model/System_and_Evaluate
  python inference.py
  ```
  - To run Text-to-Text (VI > EN) program
  ```bash
  cd Build_model/System_and_Evaluate
  python inferencevien.py
  ```
  - To run Speech-to-Speech (EN <> VI) program
  ```bash
  cd Build_model
  python s2s.py
  ```
___

--- Evaluation ---
- **BLEU Score (EN > VI)**: 22.42
- **BLEU Score (VI > EN)**: 21.76
- **BLEU Score (S2S) (EN > VI)**: 11.06
- **BLEU Score (S2S) (VI > EN)**: 9.65
___

**NOTE**
You need to download "ffmpeg.exe" and place it in the "Build_model" folder.