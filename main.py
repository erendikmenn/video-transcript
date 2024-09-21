import os
import whisper
import yt_dlp
import ffmpeg
import numpy as np
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Whisper modelini yükleyin
model = whisper.load_model("tiny")  # veya "small"

def format_timestamp(seconds):
    return f"{int(seconds // 3600):02d}:{int((seconds % 3600) // 60):02d}:{int(seconds % 60):02d}"

def get_audio_from_url(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': '-',
        'quiet': True,
        'logtostderr': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        url = info['url']
        duration = info.get('duration')
        
    out, _ = (
        ffmpeg
        .input(url)
        .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar='16k')
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0, duration

def process_video(url):
    transcript_path = 'transcript.txt'

    try:
        print("Videodan ses alınıyor...")
        audio, duration = get_audio_from_url(url)

        print("Transkript oluşturuluyor...")
        start_time = time.time()
        
        # Transkripsiyon işlemi için ilerleme çubuğu
        with tqdm(total=100, desc="İşlem İlerlemesi", bar_format="{l_bar}{bar} [ Tahmini kalan süre: {remaining} ]") as pbar:
            result = model.transcribe(audio, language="en", verbose=False)
            pbar.update(100)  # İşlem tamamlandığında çubuğu %100'e getir

        end_time = time.time()
        process_duration = end_time - start_time

        # Transkript dosyasını kaydetme
        with open(transcript_path, 'w', encoding='utf-8') as f:
            for segment in result['segments']:
                f.write(f"[{format_timestamp(segment['start'])}] {segment['text']}\n")

        print(f"\nTranskript başarıyla oluşturuldu: {transcript_path}")
        print(f"İşlem süresi: {format_timestamp(process_duration)}")

    except Exception as e:
        print(f"Bir hata oluştu: {str(e)}")

    print("İşlem tamamlandı.")

# Kullanım örneği
video_url = input("Lütfen video URL'sini girin: ")
process_video(video_url)