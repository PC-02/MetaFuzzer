# coding=utf8

import glob
import keras
import math
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import time
import subprocess
from sklearn.metrics import jaccard_score
import socket
import sys

tf.compat.v1.disable_eager_execution()

MAX_FILE_SIZE = 10000
MAX_BITMAP_SIZE = 0
SPLIT_RATIO = 0
HOST = '127.0.0.1'
PORT = 12014

round_cnt = 0

target_program_path = ""
target_dir_path = ""
dir_path = ""
file_list = ""

print("Initialization Complete")


def File_Read(path):

    global MAX_FILE_SIZE

    buf = open(path, 'rb').read()
    buf = buf + (MAX_FILE_SIZE - len(buf)) * b'\x00'
    # read file and add padding if less than max file size

    vecFile = np.frombuffer(buf, dtype=np.uint8)
    vecFile = vecFile.reshape(1, MAX_FILE_SIZE)
    # convert bytes into numpy array

    return vecFile


def Create_Bitmap(bitmaps_created):

    global MAX_FILE_SIZE
    global MAX_BITMAP_SIZE
    global SPLIT_RATIO

    data = pd.DataFrame(columns=['Input'])

    os.path.isdir("./bitmaps/") or os.makedirs("./bitmaps")
    os.path.isdir("./splice_seeds/") or os.makedirs("./splice_seeds")
    os.path.isdir("./vari_seeds/") or os.makedirs("./vari_seeds")
    os.path.isdir("./crashes/") or os.makedirs("./crashes")

    # get biggest file in list
    print(dir_path)
    fileFilter = filter(os.path.isfile, glob.glob(dir_path))
    fileMax = max(fileFilter, key=lambda x: os.stat(x).st_size)

    MAX_FILE_SIZE = os.stat(fileMax).st_size
    print(f"Max File Size: {MAX_FILE_SIZE}")

    call = subprocess.check_output
    rawBitmap = {}
    edgeList = {}

    SPLIT_RATIO = len(file_list)
    i = 0

    for file in file_list:
        out = call(
            [
                './afl-showmap', '-q', '-e', '-o',
                '/dev/stdout', '-m', '512', '-t', '500'
            ] +
            [target_program_path, '-a'] +
            [file]
        )
        tmpEdgeList = []

        if i % 100 == 0:
            print(f"{i} files have been converted to bitmaps")
        i += 1
        for line in out.splitlines():

            edge = int(line.split(b':')[0])
            tmpEdgeList.append(edge)

            # count instances of all edges found
            if edge in edgeList:
                edgeList[edge] += 1
            else:
                edgeList[edge] = 1

        # store edge list for each file
        rawBitmap[file] = np.array(tmpEdgeList)

    uniqEdges = list(edgeList.keys())
    bitmap = np.zeros(((len(file_list)), len(uniqEdges)))

    # number of edges
    MAX_BITMAP_SIZE = bitmap.shape[1]

    if not bitmaps_created:

        fileID = 0

        for file in file_list:
            fileBitmap = np.unique(rawBitmap[file])

            if fileID % 100 == 0:
                print(f"{fileID} bitmaps set")

            # get all edges for file
            for edge in fileBitmap:

                # set bitmap edge to 1 if edge covered in file
                if edge in uniqEdges:
                    bitmap[fileID][uniqEdges.index(edge)] = 1

            fileID += 1

        fileID = 0

        for file in file_list:
            # split to get relative name of file from full path
            fileName = f"./bitmaps/{file.split('/')[-1]}"
            if fileID % 100 == 0:
                print(f"{fileID} files saved")
            np.save(fileName, bitmap[fileID])
            fileID += 1

        return bitmap


def Flatten(tensor):

    # Shape tensor
    tensor = tf.reshape(tensor, (1, -1))
    # Get rid of extra dimension
    tensor = tf.squeeze(tensor)

    return tensor


def Jacc_Score(y_true, y_pred):

    # convert to tensors
    y_true = tf.round(y_true)
    y_pred = tf.round(y_pred)

    # Flatten tensors to 1D
    yTrue = tf.cast(Flatten(y_true), dtype=tf.int64)
    yPred = tf.cast(Flatten(y_pred), dtype=tf.int64)

    # return tensor of jaccard score
    return tf.convert_to_tensor(jaccard_score(yTrue, yPred),
                                dtype=tf.float32)


def Jacc_Acc(y_true, y_pred):

    pred = tf.round(y_pred)

    # cast 0 and 1s to True and False
    boolTrue = tf.cast(y_true, tf.bool)
    boolPred = tf.cast(pred, tf.bool)

    # Count wrong predictions using xor operationn
    wrongNum = tf.reduce_sum(
                    tf.cast(tf.math.logical_xor(boolTrue, boolPred),
                            dtype=tf.float32)
                )

    # Count right predictions for only 1 using and operation
    rightOneNum = tf.reduce_sum(
                tf.cast(tf.logical_and(boolTrue, boolPred),
                        dtype=tf.float32)
            )

    # Calculate Jaccard similarity coefficient
    jacc = tf.divide(rightOneNum, tf.add(rightOneNum, wrongNum))

    return keras.backend.mean(jacc)


# https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
class LossHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(Step_Decay(len(self.losses)))


def Step_Decay(epoch):
    # decaying loss rate
    initialRate = 0.0001
    drop = 0.5
    epochsDrop = 10.0
    lrate = initialRate * math.pow(drop, math.floor((1+epoch)/epochsDrop))
    return lrate


def Train_Generator(batch_size):

    global file_list

    while 1:
        np.random.shuffle(file_list)
        # load a batch of training data
        for i in range(0, SPLIT_RATIO, batch_size):

            # load full batch
            if (i + batch_size) > SPLIT_RATIO:
                x, y = Generate_Training_Data(i, SPLIT_RATIO)
                # print(x, y)
                x = x.astype('float32')

            # load remaining data for last batch
            else:
                x, y = Generate_Training_Data(i, i + batch_size)
                x = x.astype('float32')

            yield (x, y)


# splice two seeds to a new seed
def Splice_Seed(fl1, fl2, idxx):
    tmp1 = open(fl1, 'rb').read()
    ret = 1
    randd = fl2
    while ret == 1:
        tmp2 = open(randd, 'rb').read()
        if len(tmp1) >= len(tmp2):
            lenn = len(tmp2)
            head = tmp2
            tail = tmp1
        else:
            lenn = len(tmp1)
            head = tmp1
            tail = tmp2
        f_diff = 0
        l_diff = 0
        for i in range(lenn):
            if tmp1[i] != tmp2[i]:
                f_diff = i
                break
        for i in reversed(range(lenn)):
            if tmp1[i] != tmp2[i]:
                l_diff = i
                break
        if f_diff >= 0 and l_diff > 0 and (l_diff - f_diff) >= 2:
            splice_at = f_diff + random.randint(1, l_diff - f_diff - 1)
            head = list(head)
            tail = list(tail)
            tail[:splice_at] = head[:splice_at]
            with open('./splice_seeds/tmp_' + str(idxx), 'wb') as f:
                f.write(bytearray(tail))
            ret = 0
        print(f_diff, l_diff)
        randd = random.choice(seed_list)


# training data generator
def Generate_Training_Data(lb, ub):

    # initialise input and output based on upper and lower bound
    seed = np.zeros((ub - lb, MAX_FILE_SIZE))
    bitmap = np.zeros((ub - lb, MAX_BITMAP_SIZE))

    # convert files into vector representation
    for i in range(lb, ub):
        seed[i - lb] = File_Read(file_list[i])

    # read converted bitmaps
    for i in range(lb, ub):
        file_name = "./bitmaps/" + file_list[i].split('/')[-1] + ".npy"
        bitmap[i - lb] = np.load(file_name)

    return seed, bitmap


def Train(model):

    # use custom depriciating loss
    loss_history = LossHistory()
    lrate = keras.callbacks.LearningRateScheduler(Step_Decay)
    callbacks_list = [loss_history, lrate]

    # train model for batch size of 16 using generator
    model.fit(Train_Generator(16), steps_per_epoch=(SPLIT_RATIO // 16),
              epochs=10, verbose=1, callbacks=callbacks_list)

    # Save model and weights
    model.save_weights("nn_weights.h5")


def Train_LSTM(model):

    # use custom depriciating loss
    loss_history = LossHistory()
    lrate = keras.callbacks.LearningRateScheduler(Step_Decay)
    callbacks_list = [loss_history, lrate]

    x, y = Generate_Training_Data(0, SPLIT_RATIO)

    samples = 16

    x = x.reshape(x.shape[0], 1, x.shape[1])
    y = y.reshape(y.shape[0], 1, y.shape[1])

    # train model for batch size of 16 using generator
    model.fit(
        x, y, epochs=3, verbose=1, batch_size=samples, callbacks=callbacks_list
        )

    # Save model and weights
    model.save_weights("lstm_weights.h5")


def Create_Model():

    Sequential = keras.models.Sequential
    Dense = keras.layers.Dense

    model = Sequential()
    model.add(Dense(4096, input_dim=MAX_FILE_SIZE, activation='relu'))
    model.add(Dense(MAX_BITMAP_SIZE, activation='sigmoid'))

    opti = tf.keras.optimizers.Adam(learning_rate=0.0001)

    # https://en.wikipedia.org/wiki/Jaccard_index#Similarity_of_asymmetric_binary_attributes
    model.compile(
                loss='binary_crossentropy', optimizer=opti,
                metrics=[Jacc_Acc], run_eagerly=False
                )
    model.summary()

    return model


def Create_LSTM():

    Sequential = keras.models.Sequential
    LSTM = keras.layers.LSTM
    Dense = keras.layers.Dense

    model = Sequential()
    model.add(
                LSTM(
                    4096,
                    input_shape=(1, MAX_FILE_SIZE),
                    activation='tanh',
                    return_sequences=True,
                    )
             )
    model.add(Dense(MAX_BITMAP_SIZE, activation='sigmoid'))

    opti = tf.keras.optimizers.Adam(learning_rate=0.0001)

    # https://en.wikipedia.org/wiki/Jaccard_index#Similarity_of_asymmetric_binary_attributes
    model.compile(
                loss='binary_crossentropy', optimizer=opti,
                metrics=[Jacc_Acc], run_eagerly=False
                )
    model.summary()

    return model


def Gen_Gradient_Train(index, randSeeds, m, layerList, idx, sign):

    writeList = []
    global rount_cnt
    global MAX_FILE_SIZE
    # Loss Values
    loss = layerList[-2][1].output[:, index]

    # Gradient of Losses w.r to Inputs
    grad = keras.backend.gradients(loss, m.input)[0]

    # Runs graph computation giving outputs of loss layer and predictions
    # https://stackoverflow.com/questions/48142181/whats-the-purpose-of-keras-backend-function
    graphExe = keras.backend.function([m.input], [loss, grad])

    for i in range(len(randSeeds)):
        vecRandFile = File_Read(randSeeds[i])
        # print(vecRandFile.shape)
        lossVal, gradVal = graphExe([vecRandFile])

        # Sort the indexes of the grad values in desc order
        sortedGradIndexes = np.argsort(np.absolute(gradVal)).flatten()[::-1]

        if sign:
            # sorted grad values
            flatGradVals = np.absolute(gradVal).flatten()
            val = np.sign(flatGradVals[sortedGradIndexes])

        else:
            # random array of values 1 or -1 size of max file size
            val = np.random.choice([1, -1], MAX_FILE_SIZE)

        writeList.append((sortedGradIndexes, val, randSeeds[i]))

    # After the model has been trained once
    if round_cnt != 0:
        if round_cnt % 2 == 0:
            Splice_Seed(randSeeds[0], randSeeds[1], idx)

            newSeed = './splice_seeds/tmp_' + str(idx)
            vecRandFile = File_Read(newSeed)
            lossVal, gradVal = graphExe([vecRandFile])

            # Sort the indexes of the grad values in desc order
            sortedGradIndexes = np.argsort(
                np.absolute(gradVal)).flatten()[::-1]

            if sign:
                # sorted grad values
                flatGradVals = np.absolute(gradVal).flatten()
                val = np.sign(flatGradVals[sortedGradIndexes])
            else:
                # random array of values 1 or -1 size of max file size
                val = np.random.choice([1, -1], MAX_FILE_SIZE)

            writeList.append((sortedGradIndexes, val, newSeed))

    return writeList


def Gen_Gradient_Train_LSTM(index, randSeeds, m, layerList, idx, sign):

    writeList = []
    global rount_cnt
    global MAX_FILE_SIZE

    loss = layerList[-2][1].output[:, :, index]
    # output_dim = layerList[-2][1].output[:, :, :].shape
    # samples = output_dim[0]
    # timesteps = output_dim[1]
    # feat = output_dim[2]
    # samples = m.input.shape[0]
    # timesteps = m.input.shape[1]
    # feat = m.input.shape[2]

    # Gradient of Losses w.r to Inputs
    grad = keras.backend.gradients(loss, m.input)

    # Runs graph computation giving outputs of loss layer and predictions
    # https://stackoverflow.com/questions/48142181/whats-the-purpose-of-keras-backend-function
    graphExe = keras.backend.function([m.input], [loss, grad])

    for i in range(len(randSeeds)):
        vecRandFile = File_Read(randSeeds[i])
        vecRandFile = tf.convert_to_tensor(vecRandFile, dtype=float)
        vecRandFile = tf.reshape(vecRandFile, [1, 1, vecRandFile.shape[1]])

        lossVal, gradVal = graphExe(vecRandFile)

        # Sort the indexes of the grad values in desc order
        sortedGradIndexes = np.argsort(np.absolute(gradVal)).flatten()[::-1]

        if sign:
            # sorted grad values
            flatGradVals = np.absolute(gradVal).flatten()
            val = np.sign(flatGradVals[sortedGradIndexes])

        else:
            # random array of values 1 or -1 size of max file size
            val = np.random.choice([1, -1], MAX_FILE_SIZE)

        writeList.append((sortedGradIndexes, val, randSeeds[i]))

    # After the model has been trained once
    if round_cnt != 0:
        if round_cnt % 2 == 0:
            Splice_Seed(randSeeds[0], randSeeds[1], idx)

            newSeed = './splice_seeds/tmp_' + str(idx)
            vecRandFile = File_Read(newSeed)
            vecRandFile = tf.convert_to_tensor(vecRandFile, dtype=float)
            vecRandFile = tf.reshape(vecRandFile, [1, 1, vecRandFile.shape[1]])

            lossVal, gradVal = graphExe([vecRandFile])

            # Sort the indexes of the grad values in desc order
            sortedGradIndexes = np.argsort(
                np.absolute(gradVal)).flatten()[::-1]

            if sign:
                # sorted grad values
                flatGradVals = np.absolute(gradVal).flatten()
                val = np.sign(flatGradVals[sortedGradIndexes])
            else:
                # random array of values 1 or -1 size of max file size
                val = np.random.choice([1, -1], MAX_FILE_SIZE)

            writeList.append((sortedGradIndexes, val, newSeed))

    return writeList


def Gen_Mutate_Info(model, edgeNum, train, modelType):
    tmpList = []
    newSeedList = file_list
    seedLen = len(newSeedList)

    global MAX_BITMAP_SIZE

    # reuse seeds if total seeds less than edge number
    if seedLen > edgeNum:
        randSeed1 = [
            newSeedList[i] for i in np.random.choice(
                seedLen,
                edgeNum,
                replace=False
                )
            ]
        randSeed2 = [
            newSeedList[i] for i in np.random.choice(
                seedLen,
                edgeNum,
                replace=False
                )
            ]
    else:
        randSeed1 = [
            newSeedList[i] for i in np.random.choice(
                seedLen,
                edgeNum
                )
            ]
        randSeed2 = [
            newSeedList[i] for i in np.random.choice(
                seedLen,
                edgeNum
                )
            ]

    # output neurons for gradient computation
    # print(f"Max Bitmap Size: {MAX_BITMAP_SIZE}")
    # print(f"Max File Size: {MAX_FILE_SIZE}")
    outputIndices = np.random.choice(4096, edgeNum)
    layerList = [(layer.name, layer) for layer in model.layers]

    # if modelType == "LSTM":
    #     retrainVal = 50
    # else:
    #     retrainVal = 100

    with open('gradient_info_p', 'w') as f:
        for idx in range(edgeNum):
            print(f"\n***\n Edge Num: {idx} \n***\n")

            # if idx % retrainVal == 0 and idx != 0:
            #     del model
            #     keras.backend.clear_session()

            #     if modelType == "LSTM":
            #         model = Create_LSTM()
            #         model.load_weights('lstm_weights.h5')
            #     else:
            #         model = Create_Model()
            #         model.load_weights('nn_weights.h5')

            # layerList = [(layer.name, layer) for layer in model.layers]

            index = outputIndices[idx]
            randSeeds = [randSeed1[idx], randSeed2[idx]]

            if modelType == "LSTM":
                writeList = Gen_Gradient_Train_LSTM(
                    index, randSeeds, model, layerList, idx, train
                    )
            else:
                writeList = Gen_Gradient_Train(
                    index, randSeeds, model, layerList, idx, train
                    )

            tmpList.append(writeList)

            for ele in writeList:
                # stringify all data in writeList
                sortedGradIndex = [str(gradIndex) for gradIndex in ele[0]]
                randOnes = [str(ones) for ones in ele[1]]
                randSeeds = ele[2]

                # write gradient index data along with ones and file names
                f.write(
                    ",".join(sortedGradIndex) + '|'
                    + ",".join(randOnes) + '|' + randSeeds + '\n'
                       )


def Gen_Gradient(first, trained, modelType, bitmaps):

    global round_cnt

    start = time.time()
    print("Creating Bitmaps")

    Create_Bitmap(bitmaps_created=bitmaps)

    print("Bitmaps Created!")

    if modelType == "LSTM":
        model = Create_LSTM()
    else:
        model = Create_Model()

    if trained:
        print("Weights found, attaching to model...")

        if modelType == "LSTM":
            model.load_weights("./lstm_weights.h5")
        else:
            model.load_weights("./nn_weights.h5")

    else:
        print("Training model...")

        if modelType == "LSTM":
            Train_LSTM(model)
        else:
            Train(model)

    print("Model Ready!")

    if modelType == "LSTM":
        edges = 250
    else:
        edges = 250

    # Model, number of edges, and if model is being ran first time
    Gen_Mutate_Info(model, edges, first, modelType)

    round_cnt += 1
    print(f'Time Elapsed: {time.time() - start}')


def Start_Server():

    global dir_path
    global target_dir_path
    global target_program_path
    global file_list

    target_dir_path = f"{sys.argv[1]}"
    target_program_path = f"{sys.argv[2]}"
    model_type = f"{sys.argv[3]}"
    trained = sys.argv[4] == "True"
    created = sys.argv[5] == "True"
    PORT = int(sys.argv[6])

    dir_path = f"{target_dir_path}id*"
    file_list = glob.glob(dir_path)

    print("Waiting for connection...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((HOST, PORT))

    sock.listen(1)
    conn, addr = sock.accept()
    print('connected by neuzz execution moduel ' + str(addr))
    start = time.time()
    Gen_Gradient(
        first=True,
        trained=trained,
        modelType=model_type,
        bitmaps=created
    )
    conn.sendall(b"start")
    while True:
        data = conn.recv(1024)
        if not data:
            break
        else:
            Gen_Gradient(
                first=False,
                trained=trained,
                modelType=model_type,
                bitmaps=created
            )
            conn.sendall(b"start")
    conn.close()
    print(f'Total Time Elapsed: {time.time() - start}')


if __name__ == '__main__':
    Start_Server()
