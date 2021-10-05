import detechYolo

test = detechYolo.Detech("latest_med_70.pt", '0', 416, 'cpu')
test.loadModel()
test.loadData()
test.runInference()