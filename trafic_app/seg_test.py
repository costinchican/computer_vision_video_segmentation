import cv2
import numpy as np
import torch
import wget
import os
import torchvision.transforms as trf
from torchvision import models

trained_model = models.segmentation.fcn_resnet101(pretrained=True).eval()
trained_model = trained_model.cuda()

url = os.environ.get('URL')


if not os.path.exists('./custom2.mp4'):
    filename = wget.download(url, 'custom2.mp4')

capture = cv2.VideoCapture('custom2.mp4')
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
fps = (int(capture.get(cv2.CAP_PROP_FPS)))
out = cv2.VideoWriter('output_video4.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, (224, 224))

def segmentation(im, classes_no):
    r = np.zeros_like(im).astype(np.uint8)
    g = np.zeros_like(im).astype(np.uint8)
    b = np.zeros_like(im).astype(np.uint8)

    class_colors = np.array([(0, 0, 0),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (128, 0, 0), (255, 255, 255), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (0, 128, 128), (255, 255, 0), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (255, 0, 255),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    for i in range(0, classes_no):
        v = im == i
        r[v] = class_colors[i, 0]
        g[v] = class_colors[i, 1]
        b[v] = class_colors[i, 2]

    segmented_im = np.stack([r, g, b], axis=2)
    return segmented_im


for i in range(0, frame_count):
    _, img = capture.read()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    trf_list = [trf.ToPILImage(), trf.Resize(256), trf.CenterCrop(224), trf.ToTensor(),
                trf.Normalize(mean=mean, std=std)]

    transforms = trf.Compose(trf_list)
    trf_image = transforms(img)
    trf_image = trf_image.cuda()

    fcn_input = trf_image.unsqueeze(0)
    fcn_output = trained_model(fcn_input)['out']
    print(trained_model(fcn_input)['aux'])

    classes_img = torch.argmax(fcn_output.squeeze(), dim=0).detach().cpu().numpy()
    segmented_im = segmentation(classes_img, 21)

    trf_list2 = [trf.ToPILImage(), trf.Resize(256), trf.CenterCrop(224)]
    transforms2 = trf.Compose(trf_list2)
    img2 = transforms2(img)
    img2 = np.array(img2)

    final = cv2.addWeighted(img2, 1, segmented_im, 0.5, 0)
    cv2.imshow("Segmented", final)
    out.write(final)
    esc = cv2.waitKey(1)
    if esc == 27:
        break

capture.release()
out.release()
cv2.destroyAllWindows()
