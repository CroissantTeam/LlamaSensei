from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

model = load_silero_vad()
wav = read_audio('path_to_audio_file')
speech_timestamps = get_speech_timestamps(wav, model)
