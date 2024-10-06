import os
import pandas as pd

vocal_channel_map = {
    '01': 'speech',
    '02': 'song'
}

emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

emotional_intensity_map = {
    '01': 'normal',
    '02': 'strong'
}


dataset = []

audiofiles_dir = 'Audiofiles'

for filename in os.listdir(audiofiles_dir):
    if filename.endswith(".wav"):
        parts = filename[:-4].split('-')
        
        vocal_channel = vocal_channel_map.get(parts[1], 'unknown')
        emotion = emotion_map.get(parts[2], 'unknown')
        emotional_intensity = emotional_intensity_map.get(parts[3], 'unknown')
        repetition = int(parts[5])
        gender = 'male' if int(parts[6]) % 2 != 0 else 'female'
        
        data = {
            'id': filename,
            'vocal_channel': vocal_channel,
            'emotion': emotion,
            'emotional_intensity': emotional_intensity,
            'repetition': repetition,
            'gender': gender
        }
        
        dataset.append(data)

df = pd.DataFrame(dataset)

df.to_csv('audiofile_dataset.csv', index=False)
