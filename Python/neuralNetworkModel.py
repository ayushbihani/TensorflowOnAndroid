import tensorflow as tf
import numpy as np 
import logging
import os
from TextHelpers import callFunctions
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
learn = tf.contrib.learn

label_map = {"DESC":0, "ENTY":1, "HUM":2, "NUM":3, "LOC":4, "ABBR":5}
noOfClasses = 6

def getData():
    '''
    loads data from dataHelpers
    '''
    try:
        X, Y, lengthOfVector = callFunctions()
        logging.info(lengthOfVector)
        return [X,Y,lengthOfVector]
    except Exception as exception:
        logging.exception("Error in retrieving data")

def getBatch1(X,Y,i,batchSize):

    oneHotLabels = []
    for label in Y[i * batchSize:i * batchSize + batchSize]:
        y = np.zeros((6), dtype=float)
        if label == 0:
            y[0] = 1.
        elif label == 1:
            y[1] = 1.
        elif label == 2:
            y[2] = 1.
        elif label == 3:
            y[3] = 1.
        elif label == 4:
            y[4] = 1.
        elif label == 5:
            y[5] = 1.
        oneHotLabels.append(y)
    return [X[i * batchSize:i * batchSize + batchSize], np.array(oneHotLabels)]


def neuralNetwork(X, Y, lengthOfVector):
    hiddenLayer1Nodes = 800
    hiddenLayer2Nodes = 800
    nInputLayer = lengthOfVector
    trainingEpochs = 50
    learningRate = 0.01
    batchSize = 100
    displayStep = 1
    inputTensor = tf.placeholder(tf.float32,[None,nInputLayer], name="inputTensor")
    outputTensor = tf.placeholder(tf.float32, [None,noOfClasses], name="outputTensor")

    w = {
        'hiddenLayer1Weights': tf.Variable(tf.random_normal([nInputLayer, hiddenLayer1Nodes])),
        'hiddenLayer2Weights': tf.Variable(tf.random_normal([hiddenLayer1Nodes, hiddenLayer2Nodes])),
        'outputLayerWeights': tf.Variable(tf.random_normal([hiddenLayer2Nodes, noOfClasses]))
    }

    b = {
        'bias1': tf.Variable(tf.random_normal([hiddenLayer1Nodes])),
        'bias2': tf.Variable(tf.random_normal([hiddenLayer2Nodes])),
        'outputBias' : tf.Variable(tf.random_normal([noOfClasses]))
    }

    #Creating our perceptrons

    hiddenLayer1 = tf.nn.relu(tf.matmul(inputTensor,w['hiddenLayer1Weights'])+ b['bias1']) # Y=Wx+B
    hiddenLayer2 = tf.nn.relu(tf.matmul(hiddenLayer1,w['hiddenLayer2Weights'])+ b['bias2'])
    
    outLayerMultiplication = tf.matmul(hiddenLayer2, w['outputLayerWeights'])
    prediction = outLayerMultiplication + b['outputBias']
    output = tf.nn.softmax(prediction,name="output")

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=prediction, labels=outputTensor))
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learningRate).minimize(loss)
    
    saver = tf.train.Saver()    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        tf.train.write_graph(sess.graph_def, '.', 'tensorflowModel.pbtxt') 
        for epoch in range(trainingEpochs):
            avgCost = 0.
            totalBatch = int(len(X) / batchSize)
            for i in range(totalBatch):
                batchX, batchY = getBatch1(X, Y, i, batchSize)
                c, _ = sess.run([loss, optimizer], feed_dict={
                    inputTensor: batchX, outputTensor: batchY})
                avgCost += c / totalBatch
            if epoch % displayStep == 0:
                print("Epoch:", '%04d' % (epoch + 1), "loss=",
                      "{:.5f}".format(avgCost))
        print("Optimization Finished!")
        predict = tf.equal(tf.argmax(prediction, 1),
                           tf.argmax(outputTensor, 1))
        print(predict)
        accuracy = tf.reduce_mean(tf.cast(predict, "float"))
        print("Accuracy:", accuracy.eval(
            {inputTensor: xtest, outputTensor: ytest}))
        saver.save(sess,'./tensorflowModel.ckpt')                    

        tfModel = 'tensorflowModel'

        # Freeze the graph

        inputGraphPath = tfModel+'.pbtxt'
        checkpointPath = './'+tfModel+'.ckpt'
        inputSaverDefPath = ""
        inputBinary = False
        inputNode = "inputTensor"
        outputNode = "output"
        restore = "save/restore_all"
        filenameTensorName = "save/Const:0"
        outputFrozenGraph = 'frozen'+tfModel+'.pb'
        outputOptimizedGraph = 'optimized'+tfModel+'.pb'
        clearDevices = True


        freeze_graph.freeze_graph(inputGraphPath, inputSaverDefPath ,
                                inputBinary, checkpointPath, outputNode,
                                restore, filenameTensorName,
                                outputFrozenGraph, clearDevices, "")
        inputGraph = tf.GraphDef()
        with tf.gfile.Open(outputFrozenGraph, "rb") as f:
            data2read = f.read()
            inputGraph.ParseFromString(data2read)

        outputGraph = optimize_for_inference_lib.optimize_for_inference(
                inputGraph,
                [inputNode], # an array of the input node(s)
                [outputNode], # an array of output nodes
                tf.float32.as_datatype_enum)

        # Save the optimized graph

        f = tf.gfile.FastGFile(outputOptimizedGraph, "w")
        f.write(outputGraph.SerializeToString())      

X, Y, lengthOfVector= getData()
X_train = X[0:len(X)-500]
Y_train = Y[0:len(X)-500]
X_test = X[len(X)-500:len(X)]
Y_test = Y[len(X)-500:len(X)]
ckpt_dir = "./ckpt_dir"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

xtest, ytest = getBatch1(X_test, Y_test, 0, 500)
neuralNetwork(X_train, Y_train, lengthOfVector)
