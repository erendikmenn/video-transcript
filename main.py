import os
import whisper
from moviepy.editor import VideoFileClip
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import sv_ttk

# Whisper modelini yükleyin
model = whisper.load_model("base")

def format_timestamp(seconds):
    return f"{int(seconds // 3600):02d}:{int((seconds % 3600) // 60):02d}:{int(seconds % 60):02d}"

def summarize_text_with_timestamps(segments, num_sentences=5):
    sentences = [segment['text'] for segment in segments]
    timestamps = [segment['start'] for segment in segments]
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
    
    top_sentence_indices = sentence_scores.argsort()[-num_sentences:][::-1]
    
    summary = []
    for i in sorted(top_sentence_indices):
        timestamp = format_timestamp(timestamps[i])
        summary.append(f"[{timestamp}] {sentences[i]}")
    
    return "\n".join(summary)

def update_status(message):
    status_var.set(message)
    root.update_idletasks()

def update_progress(value):
    progress_var.set(value)
    root.update_idletasks()

def process_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if not video_path:
        return

    video_directory = os.path.dirname(video_path)
    video_filename = os.path.basename(video_path)
    
    audio_path = os.path.join(video_directory, 'extracted_audio.wav')
    transcript_path = os.path.join(video_directory, f'{os.path.splitext(video_filename)[0]}_transcript.txt')
    summary_path = os.path.join(video_directory, f'{os.path.splitext(video_filename)[0]}_summary.txt')

    def process():
        try:
            update_status("Videodan ses ayrıştırılıyor...")
            update_progress(20)
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)

            update_status("Transkript oluşturuluyor...")
            update_progress(40)
            result = model.transcribe(audio_path, language="en")
            segments = result['segments']

            update_status("Transkript kaydediliyor...")
            update_progress(60)
            with open(transcript_path, 'w', encoding='utf-8') as f:
                for segment in segments:
                    f.write(f"[{format_timestamp(segment['start'])}] {segment['text']}\n")

            update_status("Metin özetleniyor...")
            update_progress(80)
            summary = summarize_text_with_timestamps(segments)
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary)

            os.remove(audio_path)

            update_status("İşlem tamamlandı!")
            update_progress(100)
            messagebox.showinfo("İşlem Tamamlandı", f"Transkript ve özet başarıyla oluşturuldu.\nTranskript: {transcript_path}\nÖzet: {summary_path}")
        except Exception as e:
            messagebox.showerror("Hata", f"İşlem sırasında bir hata oluştu: {str(e)}")
        finally:
            select_button.config(state=tk.NORMAL)

    select_button.config(state=tk.DISABLED)
    threading.Thread(target=process).start()

# Ana pencereyi oluştur
root = tk.Tk()
root.title("Video Özet Asistanı")
root.geometry("800x600")
sv_ttk.set_theme("dark")

# Ana çerçeve
main_frame = ttk.Frame(root, padding="40 40 40 40")
main_frame.pack(fill=tk.BOTH, expand=True)

# Başlık
title_label = ttk.Label(main_frame, text="Video Özet Asistanı", font=("Helvetica", 28, "bold"))
title_label.pack(pady=(0, 30))

# İşlem çerçevesi
process_frame = ttk.Frame(main_frame)
process_frame.pack(fill=tk.X, pady=20)

# Dosya seçme düğmesi
select_button = ttk.Button(process_frame, text="Video Dosyası Seç", command=process_video, style="Accent.TButton", width=25)
select_button.pack(side=tk.LEFT, padx=(0, 20))

# Durum mesajı
status_var = tk.StringVar()
status_label = ttk.Label(process_frame, textvariable=status_var, font=("Helvetica", 12))
status_label.pack(side=tk.LEFT)

# İlerleme çubuğu
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(main_frame, variable=progress_var, maximum=100, length=700)
progress_bar.pack(pady=30)

# Bilgi çerçevesi
info_frame = ttk.Frame(main_frame)
info_frame.pack(fill=tk.X, pady=20)

# Bilgi etiketi
info_label = ttk.Label(info_frame, text="Video dosyanızı seçin ve özet oluşturma işlemi başlasın.", 
                       font=("Helvetica", 12), wraplength=700, justify="center")
info_label.pack()

# Ayırıcı çizgi
separator = ttk.Separator(main_frame, orient="horizontal")
separator.pack(fill=tk.X, pady=30)

# Telif hakkı bilgisi
copyright_label = ttk.Label(main_frame, text="© 2023 Video Özet Asistanı. Tüm hakları saklıdır.", 
                            font=("Helvetica", 9))
copyright_label.pack(side=tk.BOTTOM, pady=(20, 0))

# Uygulamayı başlat
root.mainloop()