import streamlit as st
from numpy import load
from numpy import expand_dims
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import math
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import ToTensor

st.header("Generate malaria detections using Faster R-CNN")
st.write("Choose any image and get corresponding detections:")

uploaded_file = st.file_uploader("Upload an image...")

def asciiart(in_f, SC, GCF,  out_f, color1='black', color2='blue', bgcolor='white'):

    # The array of ascii symbols from white to black
    chars = np.asarray(list(' .,:irs?@9B&#'))

    # Load the fonts and then get the the height and width of a typical symbol 
    # You can use different fonts here
    font = ImageFont.load_default()
    letter_width = font.getsize("x")[0]
    letter_height = font.getsize("x")[1]

    WCF = letter_height/letter_width

    #open the input file
    img = Image.open(in_f)
    model = torch.load('model_epoch_120.pth', map_location=torch.device('cpu'))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #image = torchvision.io.read_image(in_f)

    image = ToTensor()(img)#.unsqueeze(0)

    image = torch.tensor(image)
    image = image.to(device)
    image = [image]

    predictions = model(image)

    image = image[0]

    image[:] = image[:]*255

    image = image.to(torch.uint8)
    image = image.cpu()

    model.eval()

    #bbox = predictions[0]['boxes']
    #labels = predictions[0]['labels']

    # Discard detections with IoU greater or equal than 0.5 Keep the highest score
    iou_threshold = 0.5
    detected_boxes = predictions[0]['boxes']
    detected_scores = predictions[0]['scores']
    detected_labels = predictions[0]['labels']
    indices = torchvision.ops.batched_nms(boxes = detected_boxes, scores = detected_scores, idxs = detected_labels, iou_threshold = iou_threshold)

    #boxes_clean = detected_boxes
    #st.write(indices_sorted)

    boxes_clean = detected_boxes

    for count in reversed(range(len(detected_boxes))):
        if count not in indices:
            boxes_clean = torch.cat([boxes_clean[:count], boxes_clean[count+1:]])
    
    scores_clean = []
    labels_clean = []
    for indice in indices:
        scores_clean.append(detected_scores[indice])
        labels_clean.append(detected_labels[indice])

    bbox = boxes_clean
    scores = scores_clean
    labels = labels_clean
    def truncate(number, digits) -> float:
        stepper = 10.0 ** digits
        return math.trunc(stepper * number) / stepper
    # Only detections with a score greater or equal than 0.5 are considered. 
    # Anything that’s under this value is discarded.
    boxes_filtered = bbox
    scores_filtered = []
    labels_filtered = []
    score_threshold = 0.7
    for i in range(len(indices)):
        if scores[i] >= score_threshold:
            scores_filtered.append(str(truncate(scores[i].item(),3)))
            labels_filtered.append(labels[i])
    
    for i in reversed(range(len(bbox))):
        if scores[i] <= score_threshold:
            boxes_filtered = torch.cat([boxes_filtered[:i], boxes_filtered[i+1:]])

    bbox = boxes_filtered
    scores = scores_filtered
    labels = labels_filtered

    #st.write(len(detected_boxes))
    #st.write(len(indices))
    #st.write(len(bbox))
    #st.write(len(labels))

    colors = []

    leukocyteCount = 0
    malariaCount = 0
    for label in labels:
        if label == 1:
            # Leukocyte
            colors.append("blue")
            leukocyteCount += 1
        if label == 2:
            # Malaria trophozoite
            colors.append("red")
            malariaCount += 1
        if label == 3:
            # Malaria mature trophozoite
            colors.append("yellow")
            malariaCount += 1

    result = draw_bounding_boxes(image, bbox, labels=scores, colors=colors, width=3, font_size=20)

    # Save the image file
    im = torchvision.transforms.ToPILImage()(result).convert("RGB")

    #out_f = out_f.resize((1280,720))
    im.save(out_f)

    return leukocyteCount, malariaCount



def printAnalysisResults(malaria, leukocyte): 

    precisionMP = 0.871
    recallMP = 0.884
    precisionWBC = 0.977
    recallWBC = 0.997

    pd = (malaria/leukocyte)*8*10**3
    pp = ((precisionMP*recallWBC)/(precisionWBC*recallMP))*(malaria/leukocyte)*8*10**3

    st.text('# Observed Malaria parasites = ' + str(malaria))
    st.text('# Counted White Blood Cells = ' + str(leukocyte))

    st.text('PD = ' + str(math.trunc(pd)) + ' MPs/μL')
    st.text('pp = ' + str(math.trunc(pp)) + ' MPs/μL')
    
def load_image(filename, size=(512,512)):
	# load image with the preferred size
	pixels = load_img(filename, target_size=size)
	# convert to numpy array
	pixels = img_to_array(pixels)
	# scale from [0,255] to [-1,1]
	pixels = (pixels - 127.5) / 127.5
	# reshape to 1 sample
	pixels = expand_dims(pixels, 0)
	return pixels


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def imgGen2(img1):
    inputf = img1  # Input image file name

    SC = 0.1    # pixel sampling rate in width
    GCF= 2      # contrast adjustment

    leukocyteCount, malariaCount = asciiart(inputf, SC, GCF, "results.png")   #default color, black to blue
    asciiart(inputf, SC, GCF, "results_pink.png","blue","pink")
    img = Image.open(img1)
    img2 = Image.open('results.png').resize(img.size)
    #img2.save('result.png')
    #img3 = Image.open('results_pink.png').resize(img.size)
    #img3.save('resultp.png')

    return img2, leukocyteCount, malariaCount


if uploaded_file is not None:
    #src_image = load_image(uploaded_file)
    image = Image.open(uploaded_file)	
	
    st.image(uploaded_file, caption='Input Image', use_column_width=True)
    #st.write(os.listdir())
    im, leukocyteCount, malariaCount = imgGen2(uploaded_file)	
    st.image(im, caption='Malaria detections.\n Blue = Leukocyte; Red = Malaria trophozoite; Yellow = Malaria mature trophozoite', use_column_width=True)

    printAnalysisResults(malariaCount, leukocyteCount)

    st.latex(r'''
    \textup{PD}\left [ \textup{MPs} / \textup{$\mu L$} \right ] = \frac{\textup{\# MPs }}
    {\textup{\# WBCs}} \cdot  8\cdot  10^3
    ''')
    st.latex(r'''
    pp\left [ \textup{MPs} / \textup{$\mu L$} \right ] = \frac{\textup{Precision$_{\textup{MP}}$}
    \cdot \textup{Recall$_{\textup{WBC}}$}}{\textup{Precision$_{\textup{WBC}}$}\cdot 
    \textup{Recall$_{\textup{MP}}$}} \cdot \frac{\textup{\# MPs}}{\textup{\# WBCs}} \cdot  
    8\cdot  10^3
    ''')

