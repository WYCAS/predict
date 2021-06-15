from scipy.misc import toimage
import PIL.Image as pilmage
from demo import direct_img
import os
from main_event_to_img import convert_event_to_img
from keras.preprocessing import image
new_start_index=0
end_index=1
# list_percentage = [100]
split_id = 'img_pose_all_novel_split'
cwd = os.getcwd()
scene_processed_folder= os.path.join(os.path.dirname(os.getcwd()), "predict/event_data/processed/")
in_raw_events_file=os.path.join(os.path.dirname(os.getcwd()), "predict/event_data/raw_data/events.txt")
main_percentage_folder = os.path.join(scene_processed_folder, 'percentage_img')
list_event = list(open(in_raw_events_file, 'r'))
root_path = os.path.abspath(os.path.join(cwd, os.pardir))
import numpy as np
import scipy.misc as spm
from keras.models import load_model
# def dirct_predict(list_percentage, scene_processed_folder):
#     for keep_id in list_percentage:
#         # image_event_folder = '/home/anguyen/workspace/paper_src/2018.icra.event.source/event_data/processed/shapes_rotation/percentage_img/10'
#         image_event_folder = os.path.join(scene_processed_folder, 'percentage_img', str(keep_id))
#
#         out_folder = os.path.join(scene_processed_folder, 'percentage_pkl', str(keep_id))
#         if not os.path.exists(out_folder):
#             os.makedirs(out_folder)
#             # only create if not exists
#             testX=main_convert_percentage(scene_processed_folder, image_event_folder, out_folder, keep_id)
#             print testX
#             return testX
#         else:
#             testX=main_convert_percentage(scene_processed_folder, image_event_folder, out_folder, keep_id)
#             #print testX
#             return testX
#             #print 'FOLDER: ', out_folder, ' already exists. SKIP!'
# def main_convert_percentage(scene_processed_folder, img_folder, out_folder, keep_id):
#
#     test_txt_path = scene_processed_folder + 'only_event_image.txt'
#
#     list_test = list(open(test_txt_path, 'r'))
#
#     test_pkl_file = 'test.pkl'
#
#     X=write_data(list_test, img_folder, out_folder, test_pkl_file)
#     return X

def write_data(list_event):
    #list_event = list(open(in_raw_events_file, 'r'))
    #print list_event
    X = []
    # for l in list_data:
    #     l = l.rstrip('\n')
    #     gt_arr = l.split(' ')
    #     img_id = gt_arr[0]
    #     img_path = os.path.join(img_folder, img_id)
    #     X.append(img_path)
    #
    #     print '--------------------------------'
    #     print 'current l: ', l
    #     print 'img id: ', img_id
    start_index=0
    for ind in range(0,1):
        #img = image.load_img(val, target_size=(224, 224))  ## change all pixel values
        end_index=start_index+100000
        img = convert_event_to_img(start_index,end_index,list_event)
        #img.resize(224, 224)        #print img.size
        start_index=end_index
        #print ind
        img=img.reshape(224,224)

        #img.resize(224,224)#bug     bug    bug
        img = toimage(img,channel_axis=2)
        img=img.convert('RGB')


        # print img
        #img=pilmage.fromarray(np.uint8(img))
        X.append(ind)
        X[ind] = image.img_to_array(img)
        # X[ind]=img
        X[ind] /= 255.

#        print 'Process input images: ', ind, '/', len(list_data), ' -- img path: ', val

    #print 'Saving pickle file ... ', os.path.join(out_folder, pkl_file), '... done!'
    #pickle.dump((X,Y), open(os.path.join(out_folder, pkl_file), 'wb'))
    #pickle.dump(X)
    return X

def predict_pose(in_scene_id, in_net_id, in_model_file_name,list_event):
    '''
    only predict, not evaluate results (run evaluate.py script to evaluate)
    '''
    #test_txt_path = scene_processed_folder + 'only_event_image.txt'

    #list_test = list(open(test_txt_path, 'r'))

    #test_pkl_file = 'test.pkl'
    #
    # write_data(list_test, img_folder, out_folder, test_pkl_file)
    main_output = os.path.join(root_path, 'output')
    # scene_id = 'shapes_rotation'
    # scene_id = 'boxes_translation'
    scene_id = in_scene_id

    net_id = in_net_id

    model_file_name = in_model_file_name

    log_model = 'log_model'
    predition = 'prediction'

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
    print "load finished"
    testX=write_data(list_event)
    # convert to numpy array

    testX = np.array(testX)

    predicted_result = trained_model.predict(testX)
    num = 0
    print 'writing predicted result to file ...'
    fwriter = open(prediction_file, 'w')
    for p in predicted_result:
        num = num + 1
        print num
        out_line = str(p[0]) + ' ' + str(p[1]) + ' ' + str(p[2]) + ' ' + str(p[3]) + ' ' + str(p[4]) + ' ' + str(
            p[5]) + ' ' + str(p[6]) + '\n'
        print out_line
        fwriter.write(out_line)

# in_scene_id = 'shapes_6dof'
# in_net_id = 'vgg_lstm2'
# in_model_file_name = 'full_model_epoch_e0.hdf5'
# predict_pose(in_scene_id, in_net_id, in_model_file_name)
