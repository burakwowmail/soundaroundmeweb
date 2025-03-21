import sys
import tensorflow as tf
import numpy as np
import sounddevice as sd
import queue
import time
import csv
from collections import defaultdict
from datetime import datetime
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QListWidget, QTableWidget, QTableWidgetItem, QHBoxLayout, QTabWidget
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
import threading
import os
from operator import itemgetter
import rumps

# YAMNet modelini yükle
model = tf.saved_model.load('yamnet-tensorflow2-yamnet-v1')

# Sınıf isimlerini yükle
class_map_path = model.class_map_path().numpy()
class_names = []
with tf.io.gfile.GFile(class_map_path) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        class_names.append(row['display_name'])

def get_input_device():
    """En uygun mikrofon cihazını seç"""
    devices = sd.query_devices()
    print("\n🎤 Mevcut ses cihazları:")
    
    try:
        default_device = sd.query_devices(kind='input')
        default_device_index = sd.default.device[0]
    except:
        # Varsayılan cihaz bulunamazsa ilk uygun mikrofonu bul
        default_device_index = None
        default_device = None
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                default_device_index = i
                default_device = dev
                break
    
    if default_device is None:
        raise RuntimeError("Hiçbir mikrofon cihazı bulunamadı!")
    
    # Tüm giriş cihazlarını listele
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"  {i}: {dev['name']} (Kanal: {dev['max_input_channels']}, Örnek Hızı: {dev['default_samplerate']})")
    
    print(f"\n📌 Seçilen giriş cihazı: {default_device['name']}")
    return default_device_index, min(default_device['max_input_channels'], 2)  # En fazla 2 kanal

class AudioThread(QThread):
    update_signal = pyqtSignal(list, float)
    time_update_signal = pyqtSignal(float, float, float)
    DB_THRESHOLD = -47  # Added threshold as a class variable
    
    def __init__(self):
        super().__init__()
        self.q = queue.Queue()
        self.running = True
        self.block_size = 16000  # 1 saniyelik ses bloğu
        self.exposure_data = defaultdict(lambda: {'total_duration': 0, 'avg_db': 0, 'max_db': -50, 'count': 0})
        self.last_save_time = time.time()
        self.csv_file = 'noise_exposure.csv'
        self.initialize_csv()
        self.start_time = time.time()
        self.total_quiet_time = 0
        self.total_noisy_time = 0
        self.last_process_time = time.time()

    def initialize_csv(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Timestamp', 'Sound Type', 'Duration(s)', 'Decibel Level', 'Confidence'])

    def save_exposure_data(self, sound_type, duration, db_level, confidence):
        with open(self.csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                           sound_type, 
                           f"{duration:.1f}", 
                           f"{db_level:.1f}", 
                           f"{confidence:.3f}"])

        # Update running statistics
        current_data = self.exposure_data[sound_type]
        current_data['total_duration'] += duration
        current_data['avg_db'] = ((current_data['avg_db'] * current_data['count'] + db_level) / 
                                (current_data['count'] + 1))
        current_data['max_db'] = max(current_data['max_db'], db_level)
        current_data['count'] += 1

    def run(self):
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Callback status: {status}")
            self.q.put(indata.copy())

        try:
            device_index, max_channels = get_input_device()
            samplerate = 16000  # YAMNet için gerekli örnekleme hızı

            # Stream yapılandırması
            stream = sd.InputStream(
                device=device_index,
                channels=max_channels,
                samplerate=samplerate,
                callback=audio_callback,
                blocksize=self.block_size,
                dtype=np.float32
            )
            
            print(f"\n🎤 Mikrofon başlatılıyor - Cihaz: {sd.query_devices(device_index)['name']}")
            print(f"   Örnekleme Hızı: {samplerate} Hz, Kanal Sayısı: {max_channels}")
            
            with stream:
                while self.running:
                    try:
                        audio_data = self.q.get(timeout=1)
                        if audio_data is not None and len(audio_data) > 0:
                            # Stereo sesi mono'ya çevir
                            if (audio_data.shape[1] > 1):
                                audio_data = np.mean(audio_data, axis=1)
                            else:
                                audio_data = audio_data.flatten()
                            
                            # Ses sinyalinin varlığını kontrol et
                            if np.abs(audio_data).max() > 0.0001:  # Hassasiyeti artırdık
                                self.process_audio(audio_data)
                                
                    except queue.Empty:
                        continue
                    except Exception as e:
                        print(f"İşleme hatası: {str(e)}")
                        
        except Exception as e:
            print(f"\n❌ Mikrofon hatası: {str(e)}")
            print("🔄 Lütfen farklı bir ses cihazı seçin veya mikrofon izinlerini kontrol edin.")
            self.running = False

    def process_audio(self, audio_chunk):
        try:
            current_time = time.time()
            time_diff = current_time - self.last_process_time
            self.last_process_time = current_time
            
            eps = 1e-10
            audio_chunk = np.nan_to_num(audio_chunk)
            rms = np.sqrt(np.mean(np.square(audio_chunk)))
            db = 20 * np.log10(max(rms, eps))
            db = max(db, -50)

            if db > self.DB_THRESHOLD:  # Using DB_THRESHOLD
                self.total_noisy_time += time_diff
            else:
                self.total_quiet_time += time_diff

            total_elapsed = current_time - self.start_time
            self.time_update_signal.emit(total_elapsed, self.total_quiet_time, self.total_noisy_time)

            # Process audio for classification if above threshold
            if db > self.DB_THRESHOLD:
                max_abs = np.abs(audio_chunk).max()
                if max_abs > eps:
                    audio_chunk = audio_chunk.astype(np.float32) / max_abs
                    
                    if len(audio_chunk) >= 16000:
                        chunk = tf.convert_to_tensor(audio_chunk[:16000], dtype=tf.float32)
                        scores, embeddings, spectrogram = model(chunk)
                        scores = scores.numpy()[0]
                        top_5_indices = np.argsort(scores)[-5:][::-1]
                        results = []
                        highest_score = 0
                        highest_score_sound = None
                        
                        for j in top_5_indices:
                            score = float(scores[j])
                            if score > 0.3:  # Changed threshold from 0.1 to 0.3
                                sound_type = class_names[j]
                                results.append((sound_type, score))
                                # Track the highest score
                                if score > highest_score:
                                    highest_score = score
                                    highest_score_sound = sound_type
                        
                        # Save only the highest scoring sound
                        if highest_score_sound:
                            self.save_exposure_data(highest_score_sound, 1.0, db, highest_score)
                        
                        if results:
                            self.update_signal.emit(results, db)

        except Exception as e:
            print(f"Error in process_audio: {str(e)}")

    def get_sorted_exposure_data(self):
        # Convert defaultdict to list of tuples sorted by total duration
        sorted_data = [(sound_type, data['total_duration'], data['avg_db'], data['max_db'])
                      for sound_type, data in self.exposure_data.items()]
        return sorted(sorted_data, key=lambda x: x[1], reverse=True)

    def stop(self):
        self.running = False

class AudioClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.audio_thread = None
        self.setup_audio()

    def initUI(self):
        self.setWindowTitle('Live Audio Classification')
        self.setGeometry(100, 100, 800, 600)

        # Main layout
        main_layout = QVBoxLayout()
        
        # Top layout for time information
        top_layout = QHBoxLayout()
        
        # Left side - Status information
        status_layout = QVBoxLayout()
        self.status_label = QLabel("Mikrofon başlatılıyor...")
        self.label_db = QLabel("🔊 Ses Seviyesi: -- dB")
        
        for label in [self.status_label, self.label_db]:
            font = label.font()
            font.setPointSize(12)
            label.setFont(font)
            status_layout.addWidget(label)
            
        top_layout.addLayout(status_layout)
        
        # Add stretching space between left and right
        top_layout.addStretch()
        
        # Right side - Time information
        time_info_layout = QVBoxLayout()
        self.total_time_label = QLabel("⏱️ Toplam Süre: 00:00:00")
        self.quiet_time_label = QLabel("🤫 Sessiz Süre: 00:00:00")
        self.noisy_time_label = QLabel("📢 Gürültülü Süre: 00:00:00")
        
        for label in [self.total_time_label, self.quiet_time_label, self.noisy_time_label]:
            font = label.font()
            font.setPointSize(12)
            label.setFont(font)
            time_info_layout.addWidget(label)

        top_layout.addLayout(time_info_layout)
        main_layout.addLayout(top_layout)

        # Create tab widget
        self.tabs = QTabWidget()
        
        # Create two tabs
        self.tab_session = QWidget()
        self.tab_history = QWidget()
        
        # Set up session tab layout
        session_layout = QVBoxLayout()
        
        # Table for exposure data in session
        session_layout.addWidget(QLabel("Ses Maruziyeti İstatistikleri"))
        self.tableWidget = QTableWidget()
        self.tableWidget.setColumnCount(4)
        self.tableWidget.setHorizontalHeaderLabels(['Ses Türü', 'Süre (ss:dd:ss)', 'Ort. dB', 'Maks. dB'])
        session_layout.addWidget(self.tableWidget)

        # Table for real-time detections in session
        session_layout.addWidget(QLabel("Anlık Ses Tespitleri"))
        self.detectedTableWidget = QTableWidget()
        self.detectedTableWidget.setColumnCount(3)
        self.detectedTableWidget.setHorizontalHeaderLabels(['Ses Türü', 'Güven', 'dB'])
        session_layout.addWidget(self.detectedTableWidget)
        
        self.tab_session.setLayout(session_layout)

        # Set up history tab layout
        history_layout = QVBoxLayout()
        
        # Table for historical data
        self.historyTableWidget = QTableWidget()
        self.historyTableWidget.setColumnCount(5)
        self.historyTableWidget.setHorizontalHeaderLabels(['Ses Türü', 'Toplam Süre', 'Ort. dB', 'Maks. dB', 'Ort. Güven'])
        history_layout.addWidget(self.historyTableWidget)
        
        self.tab_history.setLayout(history_layout)

        # Add tabs to widget
        self.tabs.addTab(self.tab_session, "Oturum Verileri")
        self.tabs.addTab(self.tab_history, "Geçmiş İstatistikler")
        
        # Add tabs to main layout
        main_layout.addWidget(self.tabs)
        
        self.setLayout(main_layout)

        # Set up timer for updating the exposure table
        self.update_timer = self.startTimer(1000)  # Update every second
        
        # Load historical data
        self.load_historical_data()

    def format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def update_time_labels(self, total_time, quiet_time, noisy_time):
        self.total_time_label.setText(f"⏱️ Toplam Süre: {self.format_time(total_time)}")
        self.quiet_time_label.setText(f"🤫 Sessiz Süre: {self.format_time(quiet_time)}")
        self.noisy_time_label.setText(f"📢 Gürültülü Süre: {self.format_time(noisy_time)}")

    def update_exposure_table(self):
        sorted_data = self.audio_thread.get_sorted_exposure_data()
        self.tableWidget.setRowCount(len(sorted_data))
        
        for row, (sound_type, duration, avg_db, max_db) in enumerate(sorted_data):
            self.tableWidget.setItem(row, 0, QTableWidgetItem(sound_type))
            formatted_time = self.format_time(duration)
            self.tableWidget.setItem(row, 1, QTableWidgetItem(formatted_time))
            self.tableWidget.setItem(row, 2, QTableWidgetItem(f"{avg_db:.1f}"))
            self.tableWidget.setItem(row, 3, QTableWidgetItem(f"{max_db:.1f}"))
        
        self.tableWidget.resizeColumnsToContents()

    def setup_audio(self):
        self.audio_thread = AudioThread()
        self.audio_thread.update_signal.connect(self.update_ui)
        self.audio_thread.time_update_signal.connect(self.update_time_labels)
        self.audio_thread.start()
        self.status_label.setText("Mikrofon aktif - Ses bekleniyor...")

    def update_ui(self, results, db_level):
        # Cancel any existing timers
        for timer in getattr(self, '_status_timers', []):
            timer.stop()
        self._status_timers = []
            
        # Update status message
        self.status_label.setText("Ses verisi inceleniyor...")
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: self.status_label.setText("Ses verisi bekleniyor..."))
        timer.start(1500)
        self._status_timers.append(timer)
        
        # Update dB level with color coding
        if (db_level > -20):
            color = "red"
        elif (db_level > -30):
            color = "orange"
        else:
            color = "green"
            
        self.label_db.setText(f"🔊 Ses Seviyesi: {db_level:.1f} dB")
        timer2 = QTimer()
        timer2.setSingleShot(True)
        timer2.timeout.connect(lambda: self.label_db.setText("Ses verisi bekleniyor... 0 dB"))
        timer2.start(1500)
        self._status_timers.append(timer2)
        self.label_db.setStyleSheet(f"color: {color}")

        # Update detected sounds table
        self.detectedTableWidget.setRowCount(len(results))
        for row, (class_name, score) in enumerate(results):
            self.detectedTableWidget.setItem(row, 0, QTableWidgetItem(f"🔹 {class_name}"))
            self.detectedTableWidget.setItem(row, 1, QTableWidgetItem(f"{score:.3f}"))
            self.detectedTableWidget.setItem(row, 2, QTableWidgetItem(f"{db_level:.1f}"))
        
        self.detectedTableWidget.resizeColumnsToContents()

        # Reset to quiet after 1 second
        timer3 = QTimer()
        timer3.setSingleShot(True)
        timer3.timeout.connect(lambda: self.reset_to_quiet(db_level))
        timer3.start(1500)
        self._status_timers.append(timer3)

    def reset_to_quiet(self, db_level):
        self.detectedTableWidget.setRowCount(1)
        self.detectedTableWidget.setItem(0, 0, QTableWidgetItem("🤫 Sessiz"))
        self.detectedTableWidget.setItem(0, 1, QTableWidgetItem("-"))
        self.detectedTableWidget.setItem(0, 2, QTableWidgetItem(f"{db_level:.1f}"))
        self.detectedTableWidget.resizeColumnsToContents()


    def timerEvent(self, event):
        if event.timerId() == self.update_timer:
            self.update_exposure_table()

    def closeEvent(self, event):
        if self.audio_thread:
            self.audio_thread.stop()
            self.audio_thread.wait()
        event.accept()

    def load_historical_data(self):
        try:
            with open('noise_exposure.csv', 'r') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)  # Skip header row
                
                # Create a dictionary to store aggregated data
                aggregated_data = defaultdict(lambda: {
                    'total_duration': 0.0,
                    'db_levels': [],
                    'confidence_values': [],
                })
                
                # Process each record and aggregate by sound type
                for record in csv_reader:
                    timestamp, sound_type, duration, db_level, confidence = record
                    data = aggregated_data[sound_type]
                    data['total_duration'] += float(duration)
                    data['db_levels'].append(float(db_level))
                    data['confidence_values'].append(float(confidence))
                
                # Convert aggregated data to list and sort by total duration
                sorted_data = []
                for sound_type, data in aggregated_data.items():
                    avg_db = sum(data['db_levels']) / len(data['db_levels'])
                    max_db = max(data['db_levels'])
                    avg_confidence = sum(data['confidence_values']) / len(data['confidence_values'])
                    sorted_data.append((
                        sound_type,
                        data['total_duration'],
                        avg_db,
                        max_db,
                        avg_confidence
                    ))
                
                # Sort by total duration
                sorted_data.sort(key=lambda x: x[1], reverse=True)
                
                # Update table
                self.historyTableWidget.setColumnCount(5)
                self.historyTableWidget.setHorizontalHeaderLabels(['Ses Türü', 'Toplam Süre', 'Ort. dB', 'Maks. dB', 'Ort. Güven'])
                self.historyTableWidget.setRowCount(len(sorted_data))
                
                for row, (sound_type, total_duration, avg_db, max_db, avg_confidence) in enumerate(sorted_data):
                    self.historyTableWidget.setItem(row, 0, QTableWidgetItem(sound_type))
                    self.historyTableWidget.setItem(row, 1, QTableWidgetItem(self.format_time(total_duration)))
                    self.historyTableWidget.setItem(row, 2, QTableWidgetItem(f"{avg_db:.1f}"))
                    self.historyTableWidget.setItem(row, 3, QTableWidgetItem(f"{max_db:.1f}"))
                    self.historyTableWidget.setItem(row, 4, QTableWidgetItem(f"{avg_confidence:.3f}"))
                
                self.historyTableWidget.resizeColumnsToContents()
                
        except FileNotFoundError:
            print("Geçmiş veriler bulunamadı.")
        except Exception as e:
            print(f"Geçmiş veriler yüklenirken hata oluştu: {str(e)}")

def main():
    app = QApplication(sys.argv)
    ex = AudioClassifierApp()
    ex.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
