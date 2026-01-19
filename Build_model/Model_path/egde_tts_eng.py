import asyncio
import edge_tts

class VNTSS:
    def __init__(self,device="cpu"):
        print(">>>[TTS] Use Microsoft Edge TTS (onl)")
        pass
    async def genaudio(self,text, output_file,voice):
        communicate = edge_tts.Communicate(text,voice)
        await communicate.save(output_file)

    def speak(self,text,output_file = "output_vi.mp3",lang="vi"):
        if lang == "vi":
            voice = "vi-VN-NamMinhNeural"
        else:
            voice = "en-US-ChristopherNeural"
        try:
            asyncio.run(self.genaudio(text,output_file,voice))
            return output_file
        except Exception as e:
            print(f"TTS_Egde error {e}")
            return None