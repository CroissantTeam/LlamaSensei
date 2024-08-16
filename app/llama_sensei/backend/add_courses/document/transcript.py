import os

import httpx
from deepgram import DeepgramClient, FileSource, PrerecordedOptions
from dotenv import load_dotenv

load_dotenv()

DEFAULT_STT_MODEL = "nova-2"
AUDIO_SAMPLE_RATE = 48000
DEEPGRAM_API_KEY = os.getenv("DG_API_KEY")
DEEPGRAM_TIMEOUT = 900.0  # secs


class DeepgramSTTClient:
    def __init__(self, output_path) -> None:
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

        self.options = PrerecordedOptions(
            model=DEFAULT_STT_MODEL,
            sample_rate=AUDIO_SAMPLE_RATE,
            smart_format=True,
        )

    async def get_transcripts(self, audio_files):
        if len(audio_files) == 0:
            print("There is no file to transcribe")
            return
        for filename in audio_files:
            if os.path.isfile(filename):
                await self.transcribe(filename)
        return filename

    async def transcribe(self, audio_file):
        try:
            print("Connecting to Deepgram...")
            deepgram_client = DeepgramClient(DEEPGRAM_API_KEY)
            print("Connect successful!")

            with open(audio_file, "rb") as file:
                buffer_data = file.read()

            payload: FileSource = {
                "buffer": buffer_data,
            }

            print("Sending request to Deepgram...")
            r = await deepgram_client.listen.asyncrest.v("1").transcribe_file(
                payload,
                self.options,
                timeout=httpx.Timeout(DEEPGRAM_TIMEOUT, connect=10.0),
            )
            print("Received transcript results from Deepgram...")

            save_name = os.path.basename(audio_file).split(".")[0] + ".json"
            save_transcript_path = os.path.join(self.output_path, save_name)
            with open(save_transcript_path, "w") as f:
                f.write(r.to_json(indent=4))

        except Exception as e:
            print(f"Exception: {e}")
