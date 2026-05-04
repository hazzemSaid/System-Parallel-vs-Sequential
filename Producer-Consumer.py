import os
import time
import multiprocessing

from main import NUM_IMAGES, generate_dataset
from pipeline import process_image as _pipeline_process

# الإعدادات
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
NUM_CONSUMERS = 3 # عدد المستهلكين يعتمد على عدد الأنوية لديك

def producer(q, image_paths):
    """المنتج: يقرأ مسارات الصور ويضعها في الطابور"""
    for path in image_paths:
        q.put(path)
        # todo function to read file and put in queue
        print(f"[Producer] Added to queue: {os.path.basename(path)}")
        time.sleep(0.9) # محاكاة وقت قراءة الملف
    
    # إرسال إشارة توقف لكل مستهلك
    for _ in range(NUM_CONSUMERS):
        q.put(None)

def consumer(q, consumer_id):
    """المستهلك: يسحب المهام ويعالجها بشكل متوازٍ"""
    while True:
        path = q.get()
        if path is None:  # استلام إشارة النهاية
            break
        _pipeline_process(path, OUTPUT_DIR)
        print(f"[Consumer {consumer_id}] Finished: {os.path.basename(path)}")

if __name__ == "__main__":
    generate_dataset()  # توليد الصور الاصطناعية
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_paths = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith(('.jpg', '.png'))]

    # Sequential baseline for speedup/efficiency
    def seq_baseline(image_paths):
        start = time.time()
        for path in image_paths:
            _pipeline_process(path, OUTPUT_DIR)
        return time.time() - start

    seq_time = seq_baseline(image_paths)

    # إنشاء طابور مشترك لإدارة المهام[cite: 1]
    task_queue = multiprocessing.Queue()

    start_time = time.time()

    # إنشاء عملية المنتج[cite: 1]
    p = multiprocessing.Process(target=producer, args=(task_queue, image_paths))
    
    # إنشاء عمليات المستهلكين[cite: 1]
    consumers = [multiprocessing.Process(target=consumer, args=(task_queue, i)) for i in range(NUM_CONSUMERS)]

    # بدء التشغيل المتزامن (Concurrent Start)
    p.start()
    for c in consumers:
        c.start()

    # الانتظار حتى انتهاء الجميع[cite: 1]
    p.join()
    for c in consumers:
        c.join()

    par_time = time.time() - start_time

    speedup = seq_time / par_time if par_time > 0 else 0
    efficiency = (speedup / NUM_CONSUMERS) * 100

    print(f"\n{'='*45}")
    print(f"  Sequential   : {seq_time:.4f} sec")
    print(f"  Parallel     : {par_time:.4f} sec  (consumers={NUM_CONSUMERS})")
    print(f"  Speedup      : {speedup:.2f}x")
    print(f"  Efficiency   : {efficiency:.1f}%  (vs {NUM_CONSUMERS} consumers)")
    print(f"  Results saved to '{OUTPUT_DIR}/'")
    print(f"{'='*45}\n")