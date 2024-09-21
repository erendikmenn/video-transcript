import os
import whisper
from moviepy.editor import VideoFileClip
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

# NLTK'nin gerekli verilerini indirin (İngilizce için)
nltk.download('punkt')
nltk.download('stopwords')

# Videonun bulunduğu dizini ve dosya adını belirtin
video_directory = '.'  # Şu anki çalışma dizini
video_filename = 'test.mp4'
video_path = os.path.join(video_directory, video_filename)

# Ses dosyasının geçici olarak kaydedileceği yol
audio_path = os.path.join(video_directory, 'extracted_audio.wav')

# Transkript ve özet dosyalarının kaydedileceği yollar
transcript_path = os.path.join(video_directory, 'transcript.txt')
summary_path = os.path.join(video_directory, 'summary.txt')

# Whisper modelini yükleyin
model = whisper.load_model("base")

# Videodan ses ayrıştırılıyor
print("Videodan ses ayrıştırılıyor...")
video = VideoFileClip(video_path)
video.audio.write_audiofile(audio_path)

# Whisper ile ses dosyasını transkript etme (İngilizce dilinde transkript)
print("Transkript oluşturuluyor...")
result = model.transcribe(audio_path, language="en")
segments = result['segments']

def format_timestamp(seconds):
    return f"{int(seconds // 3600):02d}:{int((seconds % 3600) // 60):02d}:{int(seconds % 60):02d}"

# Transkript dosyasını kaydetme
with open(transcript_path, 'w', encoding='utf-8') as f:
    for segment in segments:
        f.write(f"[{format_timestamp(segment['start'])}] {segment['text']}\n")
print(f"Transkript başarıyla oluşturuldu: {transcript_path}")


def summarize_text_with_timestamps(segments, num_sentences=5):
    # Cümleleri ve zaman damgalarını ayır
    sentences = [segment['text'] for segment in segments]
    timestamps = [segment['start'] for segment in segments]
    
    # TF-IDF vektörleştiriciyi oluştur ve uygula
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Her cümlenin önem skorunu hesapla
    sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
    
    # En yüksek skorlu cümleleri seç
    top_sentence_indices = sentence_scores.argsort()[-num_sentences:][::-1]
    
    # Özeti oluştur ve zaman damgalarını ekle
    summary = []
    for i in sorted(top_sentence_indices):
        timestamp = format_timestamp(timestamps[i])
        summary.append(f"[{timestamp}] {sentences[i]}")
    
    return "\n".join(summary)

# Metni özetle ve summary.txt dosyasına kaydet
print("Metin özetleniyor...")
summary = summarize_text_with_timestamps(segments)
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(summary)
print(f"Özet başarıyla oluşturuldu: {summary_path}")

# Geçici ses dosyasını silme
os.remove(audio_path)

print("İşlem tamamlandı.")