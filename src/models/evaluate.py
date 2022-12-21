from audioop import reverse
from math import inf
import json
import pickle
import time
import os
import sys
from src.models.predictions.visualize import visualize
from src.utils.paths import CONFIG, OUTPUT, RAW, SRC, PUBLAYNET_TEST
sys.path.append(os.path.join(SRC,'parsers/features'))

import random

from pdf2image import convert_from_path
from PIL import Image, ImageDraw
from attrdict import AttrDict
import yaml
from src.components.graphs.labels import LableModification
from src.components.graphs.loader import Papers2Graphs
from src.utils.metrics import *
from src.utils.const import categories_names, categories_names_underscore
import numpy as np

from src.parsers.features import FeaturesREPR

def get_groundtruth_bboxs():
    out_path = SRC / 'models/predictions/groundtruth_bboxs_binary.json'

    if os.path.isfile(out_path):
        with open(out_path, 'r') as ann:
            bboxs = json.load(ann)
    
    else:
        # bboxs = {'other': dict(), 'text': dict(), 'title': dict(), 'list': dict(), 'table': dict(), 'figure': dict(),
        #         'caption': dict(), 'table-colh': dict(), 'table-sp': dict(), 'table-gcell': dict(), 'table-tcell': dict(),
        #         'table-col': dict(), 'table-row': dict()}
        bboxs = { 'group': dict() }

        with open(RAW / 'test.json', 'r') as ann:
            data = json.load(ann)
            for paper in data['papers'].keys():
                pages = data['papers'][paper]['pages']
                annotations = data['papers'][paper]['annotations']
                for idx, page in enumerate(pages): 
                    if page in ['PMC4708075_00002.pdf', 'PMC4434120_00006.pdf']: continue
                    page_anns = annotations[idx]
                    for ann in page_anns:
                        # if page in bboxs[categories_names_underscore[ann[2]].lower()].keys():
                        #     bboxs[categories_names_underscore[ann[2]].lower()][page].append([a for a in ann[0]]) # / SCALE_FACTOR
                        # else:
                        #     bboxs[categories_names_underscore[ann[2]].lower()][page] = [a for a in ann[0]] # / SCALE_FACTOR
                        cat_name = categories_names_underscore[ann[2]]
                        if cat_name not in ['TABLE-COLH', 'TABLE-SP', 'TABLE-GCELL', 'TABLE-TCELL', 'TABLE-COL', 'TABLE-ROW']:
                            if page in bboxs['group'].keys():
                                bboxs['group'][page].append([a for a in ann[0]]) # / SCALE_FACTOR
                            else:
                                bboxs['group'][page] = [[a for a in ann[0]]] # / SCALE_FACTOR
        
        with open(out_path, 'w') as f:
            json.dump(bboxs, f)
    
    return bboxs

def evaluate_map(pred_file):

    PAPER = 'PMC3443576_00003.pdf'
    visualize(PAPER)

    all_gt_boxes = get_groundtruth_bboxs()
    pred_path = SRC / f'models/predictions/{pred_file}'

    with open(pred_path) as infile:
        all_pred_boxes = json.load(infile)
    
    all_gt_boxes = {'group': {PAPER: all_gt_boxes['group'][PAPER]}}
    all_pred_boxes = {'group': {PAPER: all_pred_boxes['group'][PAPER]}}
    
    #! True to try out different scores and removing 'others' bounding boxes
    if False:
        # scores do not change map output
        scores = []
        for i in range(len(all_pred_boxes['group'][PAPER]['scores'])):
            j = i/100
            scores.append(random.uniform(0.99-2*j, 0.99-j))
        all_pred_boxes['group'][PAPER]['scores'] = scores

        # removing 'other' bounding boxes augmented by 2-3 points
        min_y1 = inf
        indeces = []
        for i, b in enumerate(all_pred_boxes['group'][PAPER]['bboxes']):
            if min_y1 >= b[1]:
                min_y1 = b[1]
                indeces.insert(0,i)
        for index in indeces:
            del all_pred_boxes['group'][PAPER]['bboxes'][index]
            del all_pred_boxes['group'][PAPER]['scores'][index]
    
    for key in all_pred_boxes.keys():
        if key not in all_gt_boxes.keys(): continue
        print(f"Evaluating {key.upper()}:")
        gt_boxes = all_gt_boxes[key]
        pred_boxes = all_pred_boxes[key]

        # Runs it for one IoU threshold
        iou_thr = 0.7
        start_time = time.time()
        data = get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=iou_thr)
        end_time = time.time()
        print('Single IoU calculation took {:.4f} secs'.format(end_time - start_time))
        print('avg precision: {:.4f}'.format(data['avg_prec']))

        start_time = time.time()
        ax = None
        avg_precs = []
        iou_thrs = []
        for idx, iou_thr in enumerate(np.linspace(0.5, 0.95, 10)):
            data = get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=iou_thr)
            avg_precs.append(data['avg_prec'])
            iou_thrs.append(iou_thr)

            precisions = data['precisions']
            recalls = data['recalls']
            ax = plot_pr_curve(
                precisions, recalls, label='{:.2f}'.format(iou_thr), color=COLORS[idx*2], ax=ax)

        # prettify for printing:
        avg_precs = [float('{:.4f}'.format(ap)) for ap in avg_precs]
        iou_thrs = [float('{:.4f}'.format(thr)) for thr in iou_thrs]
        print('map: {:.2f}'.format(100*np.mean(avg_precs)))
        print('avg precs: ', avg_precs)
        print('iou_thrs:  ', iou_thrs)
        print("")
        """
        plt.legend(loc='upper right', title='IOU Thr', frameon=True)
        for xval in np.linspace(0.0, 1.0, 11):
            plt.vlines(xval, 0.0, 1.1, color='gray', alpha=0.3, linestyles='dashed')
        end_time = time.time()
        print('\nPlotting and calculating mAP takes {:.4f} secs'.format(end_time - start_time))
        plt.show()
        """

def evaluate_doc(pred_file = OUTPUT / 'all_pred/visibility-nfeatSCIBERT_BBOX-efeat-dibi-nlay4-pmodescaled-hlaydimNone'):
    # evaluate precision and recall described in DocBank paper
    # Precision = Area GT tokens in DET tokens / Area of all DET
    # Recall = Area GT tokens in DET tokens / Area of all GT

    def real_names(label, lt):
        orig_label = lt.revert(label)
        names = list()
        for ol in orig_label:
            names.append(categories_names(ol))
        return names

    with open(CONFIG / 'graph' / "graphs.yaml") as fileobj:
        config = AttrDict(yaml.safe_load(fileobj))

    data = Papers2Graphs(config=config, test=True)

    with open(pred_file, "rb") as f:
        preds = pickle.load(f)

    gt = list()
    areas = list()
    area = lambda rect : (rect[2]- rect[0])*(rect[3]-rect[1])
    lt = LableModification(config=data.get_config())

    for idx, graph in enumerate(data.graphs):
        gt.extend(lt.convert(graph.ndata['label'].tolist()))
        areas.extend([area(b) for b in data.pages[idx]['bboxs']])
    
    gt = np.array(gt)
    preds = np.array(preds)
    areas = np.array(areas)

    # Precision
    precision = list()
    recall = list()
    f1 = list()

    names = real_names(sorted(set(gt)), lt)

    for cls in range(max(set(gt))+1):
        print(f'Evaluating {cls+1} / {max(set(gt))+1}', end='\r')
        gt_cls = np.argwhere(gt == cls)
        preds_cls = np.argwhere(preds == cls)

        # TP, FP and FN
        true_positive = np.intersect1d(gt_cls, preds_cls)
        false_positive = np.delete(preds_cls, np.argwhere(preds_cls == true_positive))
        false_negative = np.delete(gt_cls, np.argwhere(gt_cls == true_positive))

        # Areas
        true_positive_area = np.sum(areas[true_positive.tolist()])
        false_positive_area = np.sum(areas[false_positive.tolist()])
        false_negative_area = np.sum(areas[false_negative.tolist()])

        # Precision, Recall and F1-score
        p, r = true_positive_area / (true_positive_area + false_positive_area), \
                true_positive_area / (true_positive_area + false_negative_area)
        precision.append(p)
        recall.append(r)
        f1.append((2*p*r)/(p+r))
    
    print()
    print('Results:')
    for n, name in enumerate(names):
        print(f'{name}: ', f'precision {precision[n]} -', f'recall {recall[n]} -', f'f1 {f1[n]}')

    return

if __name__ == "__main__":
    
    mode = 'map'
    file = 'faster_rcnn_R_50_FPN_3x.json'
    # file = 'visib_gcnsage_bbox_scibert_thresh09.json'

    print(f"Using {file}")

    if mode == 'map':
        evaluate_map(pred_file=file)
    elif mode == 'doc':
        evaluate_doc()
    else:
        raise 'error mode'

