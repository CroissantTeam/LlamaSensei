import json

from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

model = load_silero_vad()
wav = read_audio("")
speech_timestamps = get_speech_timestamps(wav, model, return_seconds=True)
print(len(speech_timestamps))

# with open("test.json", "r") as f:
#     content = f.read()
# speech_timestamps = json.loads(content)

merge_thresh = 0.5  # seconds
duration_thresh = 0.5  # seconds
merged_timestamps = []
for i in range(1, len(speech_timestamps)):
    time_gap = speech_timestamps[i]["start"] - speech_timestamps[i - 1]["end"]
    if time_gap <= merge_thresh:
        speech_timestamps[i]["start"] = speech_timestamps[i - 1]["start"]
    else:
        segment_duration = (
            speech_timestamps[i - 1]["end"] - speech_timestamps[i - 1]["end"]
        )
        if segment_duration >= duration_thresh:
            merged_timestamps.append(speech_timestamps[i - 1])

# result = json.dumps(speech_timestamps, indent=4)
# with open("test.json", "w") as f:
#     f.write(result)

result = json.dumps(merged_timestamps, indent=4)
with open("test_merge.json", "w") as f:
    f.write(result)

print("speech_timestamps", len(speech_timestamps))
print("merged_timestamps", len(merged_timestamps))
