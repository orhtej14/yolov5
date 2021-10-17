import detechYolo

test = detechYolo.Detech("DetechModel.pt", '0', 416, 'cpu', "webcam")
test.loadModel()
test.loadData()
test.startInference()