# main.py (python example)

import os

from deepgram import DeepgramClient, FileSource, PrerecordedOptions
from dotenv import load_dotenv

load_dotenv()

# Path to the audio file
AUDIO_FILE = "spacewalk.mp3"

API_KEY = os.getenv("DG_API_KEY")


def main():
    try:
        # STEP 1 Create a Deepgram client using the API key
        dg = DeepgramClient(API_KEY)

        with open(AUDIO_FILE, "rb") as file:
            buffer_data = file.read()

        payload: FileSource = {
            "buffer": buffer_data,
        }

        # STEP 2: Configure Deepgram options for audio analysis
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
        )

        # STEP 3: Call the transcribe_file method
        r = dg.listen.prerecorded.v("1").transcribe_file(payload, options)

        # STEP 4: Print the response
        print(r.to_json(indent=4))

    except Exception as e:
        print(f"Exception: {e}")


if __name__ == "__main__":
    main()
