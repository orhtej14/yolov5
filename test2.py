import detechYolo

test = detechYolo.Detech("DetechModel.pt", '0', 416, 'cpu')
test.loadModel()
test.loadData()
test.runInference()