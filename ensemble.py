import tensorflow as tf
import pickle
import numpy as np
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
print("current pid:", os.getpid())

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[3], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[3], True)
        print("should be ok...right?")
    except RuntimeError as e:
        print(e)
else:
    print("gpu unlimited?")

def getScore(proteinName):
    testlistX, testlistY = [], []
    testfile = "TOX21/" + proteinName + '_score.smiles'
    testf = open(testfile)
    lines = testf.readlines()
    for line in lines:
        splitted = line.split("\t")
        if len(splitted[0]) >= 200:
            continue
        testlistY.append(float(splitted[1]))
    testf.close()
    #print(len(testlistY))
    return testlistY

testRes12 = [[] for i in range(12)]

testRes12[0].extend(getScore("NR-AR-LBD"))
testRes12[1].extend(getScore("NR-AR"))
testRes12[2].extend(getScore("NR-AhR"))
testRes12[3].extend(getScore("NR-Aromatase"))
testRes12[4].extend(getScore("NR-ER-LBD"))
testRes12[5].extend(getScore("NR-ER"))
testRes12[6].extend(getScore("NR-PPAR-gamma"))
testRes12[7].extend(getScore("SR-ARE"))
testRes12[8].extend(getScore("SR-ATAD5"))
testRes12[9].extend(getScore("SR-HSE"))
testRes12[10].extend(getScore("SR-MMP"))
testRes12[11].extend(getScore("SR-p53"))


def loadPickle(givenNum, logitsWhole):
    curLogits = pickle.load(open("logit"+givenNum, "rb"))#tf.float32 dtype
    if logitsWhole is None:
        logitsWhole = curLogits
    else:
        for i in range(12):
            logitsWhole[i] = tf.math.add(curLogits[i], logitsWhole[i])
    return logitsWhole

def applysigmoid(alist):
    #tmpres = [[] for i in range(12)]
    for i in range(12):
        alist[i] = tf.math.sigmoid(alist[i])
    return alist


testLogits12 = None
testLogits12 = loadPickle("0", testLogits12)
print(testLogits12[0][0])
testLogits12 = loadPickle("0", testLogits12)
print(testLogits12[0][0])
testLogits12 = loadPickle("0", testLogits12)
#print(testLogits12[0][0])
testLogits12 = applysigmoid(testLogits12)
#tfRes12
AUCFunc = tf.keras.metrics.AUC()

start = time.time()
AUCFunc.reset_states()

for i in range(12):
    AUCFunc.update_state(y_true=testRes12[i], y_pred=testLogits12[i])
print('AUC {:.4f}'.format(AUCFunc.result()))
print("time taken:", time.time() - start)
