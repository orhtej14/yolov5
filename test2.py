import detechYolo

test = detechYolo.Detech("DetechModel.pt", '0', 640, 'cpu', "webcam", [0, 1, 2, 3])
test.loadModel()
test.loadData()
test.startInference()