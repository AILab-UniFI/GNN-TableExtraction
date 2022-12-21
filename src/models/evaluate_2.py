import json
import time
import os
from src.utils.const import SCALE_FACTOR
from src.utils.metrics import *
from src.utils.paths import RAW, SRC

def get_groundtruth_bboxs():
    out_path = SRC / 'models/predictions/groundtruth_bboxs.json'

    if os.path.isfile(out_path):
        with open(out_path, 'r') as ann:
            bboxs = json.load(ann)
    
    else:
        bboxs = {'other': dict(), 'text': dict(), 'title': dict(), 'list': dict(), 'table': dict(), 'figure': dict(),
                'caption': dict(), 'table-colh': dict(), 'table-sp': dict(), 'table-gcell': dict(), 'table-tcell': dict(),
                'table-col': dict(), 'table-row': dict()}

        with open(RAW / 'test.json', 'r') as ann:
            data = json.load(ann)
            for paper in data['papers'].keys():
                pages = data['papers'][paper]['pages']
                annotations = data['papers'][paper]['annotations']
                for idx, page in enumerate(pages): 
                    page_anns = annotations[idx]
                    for ann in page_anns:
                        if page in bboxs[ann[2].lower()].keys():
                            bboxs[ann[2].lower()][page].append(ann[0]/ SCALE_FACTOR)
                        else:
                            bboxs[ann[2].lower()][page] = [ann[0]/ SCALE_FACTOR]
        
        with open(out_path, 'w') as f:
            json.dump(bboxs, f)
    
    return bboxs

def evaluate_map(pred_file = 'publaynet_pred_bboxs.json'):

    all_gt_boxes = get_groundtruth_bboxs()
    pred_path = SRC / f'models/predictions/{pred_file}'

    with open(pred_path) as infile:
        all_pred_boxes = json.load(infile)
    
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

def evaluate_doc(pred_file = 'output/all_pred/knn-nfeatBBOX-nlay3-pmodefixed-hlaydim1000'):
    # evaluate precision and recall described in DocBank paper
    # pickle
    return

if __name__ == "__main__":
    
    mode = 'map'
    file = 'ours_bboxs.json'
    # file = 'publaynet_pred_bboxs.json'

    if mode == 'map':
        evaluate_map(pred_file=file)
    elif mode == 'doc':
        evaluate_doc()
    else:
        raise 'error mode'

