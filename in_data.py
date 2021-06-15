import socket
import numpy as np
import os
import time
import random
import tensorflow as tf
import threading
from keras.models import load_model
from main_event import write_data,predict_pose
global X
global Y
# global Z
global TOTAL_NUM
TOTAL_NUM = 100000
X=[0 for x in range(0,TOTAL_NUM)]
Y=[0 for y in range(0,TOTAL_NUM)]
# Z=[0 for z in range(0,TOTAL_NUM)]
global lock_x
global lock_y
lock_x=0
lock_y=0
in_scene_id = 'shapes_6dof'
in_net_id = 'vgg_lstm2'
in_model_file_name = 'full_model_epoch_e0.hdf5'
cwd = os.getcwd()
root_path = os.path.abspath(os.path.join(cwd, os.pardir))
main_output = os.path.join(root_path, 'output')
# scene_id = 'shapes_rotation'
# scene_id = 'boxes_translation'
scene_id = in_scene_id

net_id = in_net_id

model_file_name = in_model_file_name

log_model = 'log_model'
predition = 'prediction'
split_id = 'img_pose_all_novel_split'
# save prediction result to file
predition_path = os.path.join(main_output, scene_id, net_id, split_id, predition)
prediction_file = os.path.join(predition_path, 'prediction.pre')

# load test data
# dataset_folder = '/home/anguyen/workspace/dataset/Event/processed/'
# dataset_folder = os.path.join(root_path, 'event_data', 'processed')
# data_path = os.path.join(dataset_folder, scene_id, split_id)
list_percentage = [200]

# testY = np.array(testY)
# load trained model
log_model_path = os.path.join(main_output, scene_id, net_id, split_id, log_model)
model_file = os.path.join(log_model_path, model_file_name)

trained_model = load_model(model_file)
global graph

graph = tf.get_default_graph()
print "load finished"
lock = threading.Lock()
# def fun(lock):
#     lock.acquire()
#     global lock_x
#     global lock_y
#     global X
#     global Y
#     if lock_x==0:
#         print lock_x
#         print 'predict x'
#         lock_x=1
#         #time.sleep(2)
#         testX = write_data(X)  #thread
#         testX = np.array(testX)
#         with graph.as_default():
#             predicted_result = trained_model.predict(testX)
#         print predicted_result
#
#     else:
#         print lock_x
#         print 'predict y'
#         lock_x=0
#         #time.sleep(2)
#         testY=write_data(Y)
#         testY=np.array(testY)
#         with graph.as_default():
#             predicted_result=trained_model.predict(testY)
#         print predicted_result
#     lock.release()

def fun1(lock):
    global X
    global lock_x
    lock.acquire()
    lock_x=1
    # print 'X prediction'
    # print lock_x
    #time.sleep(2)
    testX = write_data(X)  #thread
    testX = np.array(testX)
    with graph.as_default():
        predicted_result = trained_model.predict(testX)
    print predicted_result
    lock.release()

def fun2(lock):
    global Y
    global lock_x
    lock.acquire()
    lock_x=0
    # print 'Y prediction'
    # print lock_x
    #time.sleep(2)

    testY = write_data(Y)  #thread
    testY = np.array(testY)
    with graph.as_default():
        predicted_result = trained_model.predict(testY)
    print predicted_result
    lock.release()
index=0
def read_data():
    BUFSIZE = 1024
    ip_port = ('127.0.0.1', 5000)
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(ip_port)
    img = np.full((196, 256), 0.5)
    global num
    global lock_x
    num=0
    a=0
    global Y
    global X
    # global Z
    index=0
    while True:
        data, client_addr = server.recvfrom(BUFSIZE)
        #index=index+1
        #print index
        #print('GET_DATA',data)
        # data.split('\0')
        # list=data.split(' ')
        # print list[1]
        # event=data.split(' ')

        if num<TOTAL_NUM:
            if lock_x==0:
                X[num] = data
            else:
                Y[num] = data
            # Z[num]=data
        else:

            # print 'begining write'
            # prediction_file='yuhan3.txt'
            # fwriter = open(prediction_file, 'w')
            # for p in Z:
            #     #num = num + 1
            #     p=p+'\n'
            #     # out_line = str(p[0]) + ' ' + str(p[1]) + ' ' + str(p[2]) + ' ' + str(p[3])  + '\n'
            #     fwriter.write(p)


            num=0
            testX = write_data(X)  #thread
            testX = np.array(testX)
            predicted_result = trained_model.predict(testX)
            if lock_x==0:
                t1 = threading.Thread(target=fun1, args=(lock,))
                t1.start()
                lock_x=1
            else:
                t2 = threading.Thread(target=fun2, args=(lock,))
                t2.start()
                lock_x=0

        num = num + 1


        # print data[-1]
        # print data[-5:-2]
        # server.sendto(data.upper(), client_addr)
    server.close()
read_data()

