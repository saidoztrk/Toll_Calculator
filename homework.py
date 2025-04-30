import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort
from collections import defaultdict

# YouTube video URL'si
youtube_url = 'https://youtu.be/wqctLW0Hb_0?si=HCGxtHyYqO-bsSNc'

def get_video_stream(url):
    # Video akışını almak için yt-dlp kullanıyoruz
    import yt_dlp
    ydl_opts = {'format': 'best[ext=mp4]', 'quiet': True, 'noplaylist': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        for f in info_dict['formats']:
            if f['vcodec'] != 'none' and f['acodec'] != 'none' and 'url' in f:
                return f['url']
        raise ValueError("Uygun video formatı bulunamadı.")

# Video bağlantısını al
video_url = get_video_stream(youtube_url)
print(f"🎥 Video stream bağlantısı: {video_url}")

# YOLOv8 modelini yükle
model = YOLO('yolov8n.pt')  # YOLOv8 small modelini yükle

# Takip edilmesi gereken araç sınıfları
target_classes = ['car', 'motorcycle', 'bus', 'truck', 'bicycle']

# Ücretlendirme bilgileri
pricing = {'car': 15, 'motorcycle': 10, 'bus': 20, 'truck': 25, 'bicycle': 5}

# Araç takip bilgileri
vehicle_stats = {}  # {id: (class, center_x, center_y)}
vehicle_counts = defaultdict(int)
total_revenue = 0

# Deep SORT takibini başlat
tracker = Sort()

# Video kaynağını aç
cap = cv2.VideoCapture(video_url)

if not cap.isOpened():
    print("❌ Video açılamadı.")
    input("Program tamamlandı. Kapatmak için Enter tuşuna bas...")
    exit()
else:
    print("✅ Video başarıyla açıldı.")

# FPS kontrolü ve başlama noktası
fps = cap.get(cv2.CAP_PROP_FPS)
start_frame = int(fps * 1132)  # İstediğiniz başlangıç çerçevesi
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Pencereyi oluştur
cv2.namedWindow("Arac_Takip_Kopru_Ucreti", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Arac_Takip_Kopru_Ucreti", 960, 540)

fps_control = 0
rate_limit = 20

# Araçları tekrar tespit etme kontrolü
def is_duplicate_vehicle(new_center, new_class, threshold=50):
    for _, (v_class, cx, cy) in vehicle_stats.items():
        if v_class == new_class:
            dist = np.linalg.norm(np.array(new_center) - np.array((cx, cy)))
            if dist < threshold:
                return True
    return False

while True:
    fps_control += 1
    if fps_control % rate_limit != 0:
        continue

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))
    
    # YOLOv8 modelinin sonuçlarını işleme (torch.no_grad() ile bellek ve işlemci optimizasyonu)
    results = model(frame)

    detections = []
    class_map = {}

    # YOLOv8 modelinin sonuçlarını işleme
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # box.xyxy ile koordinatları alıyoruz
            conf = box.conf[0].item()  # Güven skoru
            cls = int(box.cls[0].item())  # Sınıf ID'si
            class_name = model.names[cls]  # Sınıf adı
            if class_name in target_classes:
                detections.append([x1, y1, x2, y2, conf])
                class_map[(x1, y1, x2, y2)] = class_name

    # Deep SORT için tespit edilen nesneler
    if detections:
        det_array = np.array(detections, dtype=np.float32)
        tracked_objects = tracker.update(det_array)
    else:
        tracked_objects = np.empty((0, 5))

    # Takip edilen nesneleri işleme
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        matched_class = None
        best_iou = 0

        # Sınıf eşleştirmesi için IoU hesaplama
        for box_coords, class_name in class_map.items():
            iou_x1, iou_y1, iou_x2, iou_y2 = box_coords
            xi1 = max(iou_x1, x1)
            yi1 = max(iou_y1, y1)
            xi2 = min(iou_x2, x2)
            yi2 = min(iou_y2, y2)
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            box1_area = (iou_x2 - iou_x1) * (iou_y2 - iou_y1)
            box2_area = (x2 - x1) * (y2 - y1)
            union_area = box1_area + box2_area - inter_area
            iou = inter_area / union_area if union_area != 0 else 0

            if iou > best_iou:
                best_iou = iou
                matched_class = class_name

        if matched_class:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Eğer araç daha önce ücretlendirildiyse tekrar edilmesin
            if obj_id not in vehicle_stats:
                if not is_duplicate_vehicle((center_x, center_y), matched_class):
                    vehicle_stats[obj_id] = (matched_class, center_x, center_y)
                    vehicle_counts[matched_class] += 1
                    total_revenue += pricing[matched_class]

            label = f"{matched_class} - ID:{obj_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Üst bilgi paneli
    cv2.rectangle(frame, (0, 0), (960, 60), (50, 50, 50), -1)
    summary = f"Toplam Gelir: {total_revenue} TL | " + " | ".join(
        [f"{k}: {vehicle_counts[k]}" for k in target_classes]
    )
    cv2.putText(frame, summary, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Arac_Takip_Kopru_Ucreti", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizleme ve kapanış işlemleri
cap.release()
cv2.destroyAllWindows()
