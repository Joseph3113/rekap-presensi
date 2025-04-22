import os
import cv2
import time
import json
import numpy as np
import paramiko
import csv
from flask import Flask, render_template, request, jsonify, Response, send_from_directory, redirect, url_for, flash

app = Flask(__name__)

# Set secret_key untuk sesi
app.config['SECRET_KEY'] = 'your_secret_key_here'  # Ganti dengan key yang lebih aman

# Path
TEMP_DIR = "template_wajah"
CSV_FILE = "rekap_presensi.csv"
JSON_FILE = "rekap_presensi.json"
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Konfigurasi koneksi VPS
VPS_HOST = "103.172.205.72"
VPS_PORT = 22
VPS_USERNAME = "riady"
VPS_PASSWORD = "Halo1234"
VPS_DEST_PATH = "/home/riady/presensi/rekap.csv"

# Global variables
templates = {}
attendance_log = {}
detection_start_time = {}

# Load data
def load_templates():
    templates.clear()
    for nama in os.listdir(TEMP_DIR):
        path = os.path.join(TEMP_DIR, nama, "template.jpg")
        if os.path.exists(path):
            img = cv2.imread(path, 0)
            templates[nama] = img

def load_attendance():
    global attendance_log
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r") as f:
            attendance_log = json.load(f)

def save_attendance():
    with open(JSON_FILE, "w") as f:
        json.dump(attendance_log, f, indent=4)
    save_attendance_csv()

def save_attendance_csv():
    # Save attendance to CSV
    try:
        with open(CSV_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Nama", "Waktu Kehadiran"])
            for nama, times in attendance_log.items():
                if isinstance(times, list):
                    for t in times:
                        writer.writerow([nama, t])
                else:
                    writer.writerow([nama, times])

        # Kirim file CSV ke VPS setelah menyimpan
        upload_csv_to_vps()
    except Exception as e:
        print(f"Error saving CSV: {e}")

def upload_csv_to_vps():
    try:
        # Kirim file CSV ke VPS via Paramiko
        print("Mengirim file CSV ke VPS...")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(VPS_HOST, port=VPS_PORT, username=VPS_USERNAME, password=VPS_PASSWORD)

        sftp = ssh.open_sftp()
        sftp.put(CSV_FILE, VPS_DEST_PATH)
        sftp.close()
        ssh.close()

        print("File CSV berhasil dikirim ke VPS.")
    except Exception as e:
        print(f"Gagal mengirim CSV ke VPS: {str(e)}")

# Routes
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/lihat_kehadiran')
def lihat_kehadiran():
    if not os.path.exists(JSON_FILE):
        return render_template('rekap.html', data={}, message="Tidak ada data kehadiran.")
    with open(JSON_FILE, 'r') as f:
        data = json.load(f)
    return render_template('rekap.html', data=data, message="Daftar kehadiran:")

@app.route('/capture_template', methods=['POST'])
def capture_template():
    nama = request.form['nama']
    cap = cv2.VideoCapture(1)  # Pastikan kamera yang benar
    time.sleep(2)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({"message": "Gagal mengambil gambar."})

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return jsonify({"message": "Tidak ada wajah terdeteksi."})

    x, y, w, h = faces[0]
    face_img = gray[y:y+h, x:x+w]
    os.makedirs(os.path.join(TEMP_DIR, nama), exist_ok=True)
    cv2.imwrite(os.path.join(TEMP_DIR, nama, "template.jpg"), face_img)
    load_templates()

    return jsonify({"message": f"Template wajah untuk {nama} telah disimpan."})

@app.route('/start_detection', methods=['POST'])
def start_detection():
    detection_start_time.clear()
    return jsonify({"message": "Deteksi dimulai!"})

@app.route('/view_attendance', methods=['GET'])
def view_attendance():
    if not attendance_log:
        return jsonify({"attendance": "Belum ada yang hadir."})

    lines = []
    for nama, times in attendance_log.items():
        times_list = times if isinstance(times, list) else [times]
        for t in times_list:
            lines.append(f"{nama}: {t}")

    return jsonify({"attendance": "\n".join(lines)})

@app.route('/attendance_log')
def download_log():
    return send_from_directory(os.getcwd(), JSON_FILE)

@app.route('/hapus_rekap', methods=['POST'])
def hapus_rekap():
    global attendance_log
    attendance_log.clear()  # Menghapus data kehadiran yang ada di memori

    # Kosongkan file JSON (hapus data saja, tidak hapus file)
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'w') as f:
            json.dump({}, f)

    # Kosongkan file CSV (hapus data saja, tidak hapus file)
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Nama", "Waktu Kehadiran"])  # Hanya menulis header tanpa data

    # Menghapus data di VPS juga
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(VPS_HOST, port=VPS_PORT, username=VPS_USERNAME, password=VPS_PASSWORD)

        sftp = ssh.open_sftp()
        sftp.put(CSV_FILE, VPS_DEST_PATH)  
        sftp.close()
        ssh.close()

        print("File CSV berhasil dikirim ke VPS.")
    except Exception as e:
        print(f"Gagal mengirim CSV ke VPS: {str(e)}")

    flash("Data kehadiran berhasil dihapus.", "success")
    return redirect(url_for('lihat_kehadiran'))

def gen_frames():
    cap = cv2.VideoCapture(1)  
    if not cap.isOpened():
        raise RuntimeError("Kamera tidak bisa dibuka.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            for nama, template in templates.items():
                try:
                    resized = cv2.resize(face_roi, (template.shape[1], template.shape[0]))
                    diff = cv2.absdiff(resized, template)
                    score = np.mean(diff)

                    if score < 40:
                        if nama not in detection_start_time:
                            detection_start_time[nama] = time.time()
                        elif time.time() - detection_start_time[nama] > 2:
                            if nama not in attendance_log:
                                attendance_log[nama] = []

                            # Tambahkan waktu baru jika belum terlalu dekat dari deteksi sebelumnya
                            if len(attendance_log[nama]) == 0 or \
                               (time.time() - time.mktime(time.strptime(attendance_log[nama][-1], "%Y-%m-%d %H:%M:%S")) > 10):
                                attendance_log[nama].append(time.strftime("%Y-%m-%d %H:%M:%S"))
                                save_attendance()

                            cv2.putText(frame, f"Hadir: {nama}", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            break
                except Exception:
                    continue

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    os.makedirs(TEMP_DIR, exist_ok=True)
    load_attendance()
    load_templates()
    app.run(host='0.0.0.0', port=5000, debug=True)
