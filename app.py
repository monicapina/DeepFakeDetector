import sys
from scipy.special import expit
from architectures import fornet, weights
from isplutils import utils
from blazeface import FaceExtractor, BlazeFace, VideoReader
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import torch
from torch.utils.model_zoo import load_url
from PIL import Image
import matplotlib.pyplot as plt
from flask_cors import CORS, cross_origin
from flask import jsonify
from twitterdl import TwitterDownloader
import os
import random
app = Flask(__name__)


# """
# Choose an architecture between
# - EfficientNetB4
# - EfficientNetB4ST
# - EfficientNetAutoAttB4
# - EfficientNetAutoAttB4ST
# - Xception
# """
# net_model = 'EfficientNetAutoAttB4'

# """
# Choose a training dataset between
# - DFDC
# - FFPP
# """
# train_db = 'DFDC'

# device = torch.device(
#     'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# face_policy = 'scale'
# face_size = 224
# frames_per_video = 32
# model_url = weights.weight_url['{:s}_{:s}'.format(net_model, train_db)]
# net = getattr(fornet, net_model)().eval().to(device)
# net.load_state_dict(load_url(model_url, map_location=device, check_hash=True))
# facedet = BlazeFace().to(device)
# facedet.load_weights("./blazeface/blazeface.pth")
# facedet.load_anchors("./blazeface/anchors.npy")
# face_extractor = FaceExtractor(facedet=facedet)
# videoreader = VideoReader(verbose=False)


# def video_read_fn(x): return videoreader.read_frames(
#     x, num_frames=frames_per_video)


# transf = utils.get_transformer(
#     face_policy, face_size, net.get_normalizer(), train=False)


def predictImage(image):
    image = Image.open(image)
    image = face_extractor.process_image(img=image)
    # take the face with the highest confidence score found by BlazeFace
    image = image['faces'][0]

    faces_t = torch.stack([transf(image=im)['image']
                           for im in [image]])

    with torch.no_grad():
        faces_pred = torch.sigmoid(
            net(faces_t.to(device))).cpu().numpy().flatten()

    d = {'result': str(faces_pred[0])}
    #print(round(faces_pred[0]))
    return jsonify(d)


"""
Choose an architecture between
- EfficientNetB4
- EfficientNetB4ST
- EfficientNetAutoAttB4
- EfficientNetAutoAttB4ST
- Xception
"""
net_model = 'EfficientNetAutoAttB4ST'

"""
Choose a training dataset between
- DFDC
- FFPP
"""
train_db = 'FFPP'

device = torch.device(
    'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
face_policy = 'scale'
face_size = 224
frames_per_video = model_url = weights.weight_url['{:s}_{:s}'.format(net_model, train_db)]


net = getattr(fornet, net_model)().eval().to(device)
net.load_state_dict(load_url(model_url, map_location=device, check_hash=True))

facedet = BlazeFace().to(device)
facedet.load_weights("./blazeface/blazeface.pth")
facedet.load_anchors("./blazeface/anchors.npy")
videoreader = VideoReader(verbose=False)
transf = utils.get_transformer(
    face_policy, face_size, net.get_normalizer(), train=False)

print('Model loaded')
def video_read_fn(x): return videoreader.read_frames(
    x, num_frames=32)


face_extractor = FaceExtractor(video_read_fn=video_read_fn, facedet=facedet)


def predictVideo(video):
    vid_real_faces = face_extractor.process_video(video)
    #print(vid_real_faces)
    im_real_face = vid_real_faces[0]['faces'][0]
    #print(im_real_face)
    faces_real_t = torch.stack([transf(image=frame['faces'][0])[
                               'image'] for frame in vid_real_faces if len(frame['faces'])])
    #print(faces_real_t.shape)
    with torch.no_grad():
        faces_real_pred = torch.sigmoid(net(faces_real_t.to(device))).cpu().numpy().flatten()
        #print(faces_real_pred)
        
    d = {'result': str(faces_real_pred.mean())}
    
    return d['result']
def classify_binary(predictions, threshold=0.5):
    """
    Classify float predictions for binary classification.

    Parameters:
    - predictions: list of float predictions.
    - threshold: float, threshold to decide between class 0 and class 1.

    Returns:
    - A list of 0s and 1s representing the predicted class.
    """
    return [1 if pred >= threshold else 0 for pred in predictions]



if __name__ == "__main__":
    #print('Starting server')
   
    #True=0 False=1
    
    #predictVideo("/home/adaptai/dataset/FaceForensics++/original_sequences/youtube/c23/videos/019.mp4")
    #predictVideo("/home/adaptai/dataset/Fakevaceleb/FakeAVCeleb_v1.2/FakeVideo-FakeAudio/Caucasian (American)/women/id03781/00113_id00431_wavtolip.mp4")
    #predictVideo("//home/adaptai/dataset/Fakevaceleb/FakeAVCeleb_v1.2/FakeVideo-RealAudio/African/men/id00076/00109_3.mp4") #Fake/true
    #predictVideo("/home/adaptai/dataset/Fakevaceleb/FakeAVCeleb_v1.2/FakeVideo-RealAudio/African/men/id00076/00109_11.mp4") #fAKE/fake
    #predictVideo("/home/adaptai/dataset/Fakevaceleb/FakeAVCeleb_v1.2/FakeVideo-RealAudio/African/men/id00076/00109_id04727_AAnyugIAXao.mp4") #Fake/true
    #predictVideo("/media/adaptai/T7 Shield/MONICA/dataset/FaceForensics/FaceForensics_compressed/test/altered/7sSPBQvxImQ_0_z3NNFRvZgNs_2.avi") #True/true0
    #predictVideo("/media/adaptai/T7 Shield/MONICA/dataset/FaceForensics/FaceForensics_compressed/test/original/7sSPBQvxImQ_0_z3NNFRvZgNs_2.avi") #True/True
    
    #predictVideo("/media/adaptai/T7 Shield/MONICA/dataset/FaceForensics/FaceForensics_compressed/test/altered/wnx2fsN9WP0_1_wbLsNxqHyeA_1.avi") #False/true0
    
    
    #predictImage("output/i/GHJMM8JWsAAyoP9.jpeg")
    
    #folder_deepfake = '/media/adaptai/T7 Shield/MONICA/dataset/FaceForensics/FaceForensics_compressed/test/altered/'
    
    folder_deepfake="/media/adaptai/T7 Shield/MONICA/dataset/Fakevaceleb/FakeAVCeleb_v1.2/FakeVideo-FakeAudio/"
    folder_original= '/media/adaptai/T7 Shield/MONICA/dataset/FaceForensics/FaceForensics_compressed/test/original/'
            
    success=0
    iteration=0
    total=0
    for raiz, dirs, archivos in os.walk(folder_deepfake):
        for nombre_archivo in archivos:
            ruta_completa = os.path.join(raiz, nombre_archivo)
            #print(ruta_completa)
            result=predictVideo(ruta_completa)
            
            classified_predictions = classify_binary([float(result)])
            #print(classified_predictions[0])
            if classified_predictions[0]==1:
                print("### iteration",iteration,": Deepfake")
                success+=1
            else: 
                print("### iteration",iteration,": Real")
            iteration+=1

    print("Success: ",success/iteration*100)

