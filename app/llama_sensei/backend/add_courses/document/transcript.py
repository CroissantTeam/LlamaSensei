# main.py (python example)

import os

import httpx
from deepgram import DeepgramClient, FileSource, PrerecordedOptions
from dotenv import load_dotenv

load_dotenv()

# Path to the audio file
AUDIO_FILE = ""

API_KEY = os.getenv("DG_API_KEY")
print("API_KEY", API_KEY)


def main():
    try:
        # STEP 1 Create a Deepgram client using the API key
        deepgram_client = DeepgramClient(API_KEY)

        with open(AUDIO_FILE, "rb") as file:
            buffer_data = file.read()

        payload: FileSource = {
            "buffer": buffer_data,
        }

        # STEP 2: Configure Deepgram options for audio analysis
        options = PrerecordedOptions(
            model="nova-2", smart_format=True, sample_rate=48000
        )

        # STEP 3: Call the transcribe_file method
        r = deepgram_client.listen.prerecorded.v("1").transcribe_file(
            payload, options, timeout=httpx.Timeout(900.0, connect=10.0)
        )

        # STEP 4: Print the response
        with open("stanford_cs229_l1.json", "w") as f:
            f.write(r.to_json(indent=4))

    except Exception as e:
        print(f"Exception: {e}")


if __name__ == "__main__":
    main()
