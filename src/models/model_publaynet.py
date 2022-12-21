import layoutparser as lp
from pdf2image import convert_from_path
from PIL import ImageDraw
import os
from tqdm import tqdm
import json

from src.utils.paths import PUBLAYNET_TEST, RAW, SRC

COLORS = {'TEXT': (255,  0, 0), 'TITLE': (0, 255, 0), 'LIST': (0, 0, 255), 'TABLE': (255, 255, 0), 'FIGURE': (255,  0, 255)}

def get_names():
    names = list()
    with open(RAW / 'test.json', 'r') as ann:
        data = json.load(ann)
        j = 0
        for paper in data['papers'].keys():
            pages = data['papers'][paper]['pages']
            for page in pages: names.append(page)
    return names

model_name = 'faster_rcnn_R_50_FPN_3x'
# model_name = 'mask_rcnn_X_101_32x8d_FPN_3x'
# PubLayNey pretrained model
out_path = SRC / f'models/predictions/{model_name}.json'

if os.path.isfile(out_path):
    print("Already done -> ", out_path)

else:
    model = lp.Detectron2LayoutModel(
                config_path =f'lp://PubLayNet/{model_name}/config', # In model catalog
                label_map   ={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}, # In model`label_map`
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8] # Optional
            )

    # pred_bboxs = {'text': dict(), 'title': dict(), 'list': dict(), 'table': dict(), 'figure': dict()}
    pred_bboxs = {'group': dict()}
    to_list = lambda rect: [rect.x_1, rect.y_1, rect.x_2, rect.y_2]

    for pdf_name in tqdm(get_names(), desc='detecting on images'):
        try:
            image = convert_from_path(PUBLAYNET_TEST / pdf_name)[0]
        except: continue
        layout = model.detect(image)
        for l in layout:
            # if pdf_name in pred_bboxs[l.type.lower()].keys():
            #     pred_bboxs[l.type.lower()][pdf_name]['bboxes'].append(to_list(l.block))
            #     pred_bboxs[l.type.lower()][pdf_name]['scores'].append(l.score)
            # else:
            #     pred_bboxs[l.type.lower()][pdf_name] = dict()
            #     pred_bboxs[l.type.lower()][pdf_name]['bboxes'] = [to_list(l.block)]
            #     pred_bboxs[l.type.lower()][pdf_name]['scores'] = [l.score]
            if pdf_name in pred_bboxs['group'].keys():
                pred_bboxs['group'][pdf_name]['bboxes'].append(to_list(l.block))
                pred_bboxs['group'][pdf_name]['scores'].append(l.score)
            else:
                pred_bboxs['group'][pdf_name] = dict()
                pred_bboxs['group'][pdf_name]['bboxes'] = [to_list(l.block)]
                pred_bboxs['group'][pdf_name]['scores'] = [l.score]

        # blocks = [[l.block, l.type, l.score] for l in layout]
        #print("Document Layout Analysis : PubLayNet")
        #print(blocks)

    with open(out_path, 'w') as f:
        json.dump(pred_bboxs, f)

"""# TableBank pretrained model

model = lp.Detectron2LayoutModel(
            config_path ='lp://TableBank/faster_rcnn_R_101_FPN_3x/config', # In model catalog
            label_map   ={0: "Table"}, # In model`label_map`
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8] # Optional
        )

image = convert_from_path(PUBLAYNET_TEST / "PMC1087887_00003.pdf")[0]
layout = model.detect(image)
blocks = [[l.block, l.type, l.score] for l in layout]
print("Table Detection : Table Bank")
print(blocks)

draw = ImageDraw.Draw(image)
for b in blocks:
    draw.rectangle([(b[0].x_1, b[0].y_1), (b[0].x_2, b[0].y_2)], outline=COLORS[b[1].upper()], width=3)
image.save('tabbank_prova.png')"""