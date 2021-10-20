import detechYolo
import threading

test = detechYolo.Detech("best.pt", "0", 640, 'cpu', "webcam", [0, 1, 2, 3])
test.loadModel()
test.loadData()

# thread1 = threading.Thread(target=test.loadModel, daemon=True)
# thread2 = threading.Thread(target=test.loadData, daemon=True)
thread3 = threading.Thread(target=test.startInference, daemon=False)
# thread1.start()
# thread2.start()
thread3.start()