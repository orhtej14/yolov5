import detechYolo

test = detechYolo.Detech("epoch_250.pt", "0", 640, 'cpu', "webcam")
test.show_res = True
# test.startInference()
test.loadModel()
test.loadData()
test.isDetecting = True
test.runInference()