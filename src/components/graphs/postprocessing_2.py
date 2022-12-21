from email import header
import json
import pickle
from src.components.graphs.utils import intersect
from src.components.graphs.labels import LableModification
from src.components.graphs.loader import Papers2Graphs
from src.components.graphs.utils import parse_args_ModelPredict
import fitz
from pdf2image import convert_from_path
from PIL import ImageDraw, ImageFont
from src.utils.const import categories_colors, categories_names
from src.utils.const import Categories_names as cm
from src.utils.const import SCALE_FACTOR as sf
import torch
from src.utils.fs import create_folder_if_not_exists
from tqdm import tqdm

from src.utils.paths import CONFIG, OUTPUT, PUBLAYNET_TEST, PUBLAYNET_TRAIN, SRC
FONT = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMono.ttf', size=30)

def get_tables_bbox(blocks, block_labels):
    #! add table counting
    block_tables = {
        cm.TABLE_COLH.value: [i for i, h in enumerate(block_labels) if h == cm.TABLE_COLH.value],
        cm.TABLE_SP.value: [i for i, h in enumerate(block_labels) if h == cm.TABLE_SP.value],
        cm.TABLE_TCELL.value: [i for i, h in enumerate(block_labels) if h == cm.TABLE_TCELL.value]
    }

    intervals = list()
    
    #! find vertical intersecting table blocks (groups are columns)
    headers = []
    for cls, block_list in block_tables.items():
        if block_list:
            cells_vertical_groups = \
                {0: {'int': [blocks[block_list[0]][0], blocks[block_list[0]][2]], 'blk': [], 'y_centers': [], 'l': cm.TABLE_TCELL.value}}
            for block_c in block_list:
                block = blocks[block_c]
                not_found = True
                for group in cells_vertical_groups.keys():
                    interval, blks, ycs, _ = cells_vertical_groups[group].values()
                    if interval[0] <= block[2] and interval[1] >= block[0]:
                        interval = [min(interval[0], block[0]), max(block[2], interval[1])]
                        y_center = (block[3] + block[1])/2

                        for i, y in enumerate(ycs):
                            if y < y_center:
                                continue
                            else:
                                blks.insert(i, block) 
                                ycs.insert(i, y_center)
                                not_found = False
                                break
                        
                        if not_found:
                            blks.append(block)
                            ycs.append(y_center)
                            not_found = False
                        break

                if not_found:
                    cells_vertical_groups[len(cells_vertical_groups.keys())] = \
                    {'int': [block[0], block[2]], 'blk': [block], 'y_centers': [(block[3] + block[1])/2], 'l': cm.TABLE_TCELL.value}

            #! split vertical groups into more tables (if any)
            tolerance = 2
            groups_splits = []
            for _, group in cells_vertical_groups.items():
                splits = []
                count = 0
                interval = group['int']
                for i, block in enumerate(blocks):
                    if interval[0] <= block[2] and interval[1] >= block[0]:
                        ycb = (block[3] + block[1]) / 2
                        for j, yct in enumerate(group['y_centers']):
                            if ycb < yct: break
                            if j+1 == len(group['y_centers']): break
                            if yct < ycb <  group['y_centers'][j+1]:
                                count += 1
                                if count == tolerance:
                                    splits.append(j+1)
                                break
                groups_splits.append(splits)
            
            #! merge blocks and create tables
            merge = lambda gblk : [min([x[0] for x in gblk]), min([x[1] for x in gblk]), max([x[2] for x in gblk]), max([x[3] for x in gblk])]
            for key, group in cells_vertical_groups.items():
                old_blocks = group['blk']
                splits = groups_splits[key]
                split_start = 0
                for i, split_mid in enumerate(splits):
                    group_blocks = old_blocks[split_start:split_mid]
                    if group_blocks:
                        new_block = merge(group_blocks)
                        blocks.append(new_block)
                        block_labels.append(cls)
                        if cls == cm.TABLE_COLH.value or cls == cm.TABLE_SP.value: headers.append(new_block)
                    if i+1 == len(groups_splits[key]):
                        group_blocks = old_blocks[split_mid:]
                        if group_blocks:
                            new_block = merge(group_blocks)
                            blocks.append(new_block)
                            block_labels.append(cls)
                            if cls == cm.TABLE_COLH.value or cls == cm.TABLE_SP.value: headers.append(new_block)              
                if not splits:
                    new_block = merge(old_blocks)
                    blocks.append(new_block)
                    block_labels.append(cls)
                    if cls == cm.TABLE_COLH.value or cls == cm.TABLE_SP.value: headers.append(new_block)

            if cls == cm.TABLE_TCELL.value: intervals.extend([val['int'] for val in cells_vertical_groups.values()])

    #! remove old blocks that have been merged into tables
    remove_block = list()
    for item in block_tables.values():
        remove_block.extend(item)
    for index in sorted(remove_block, reverse=True):
        del blocks[index]
        del block_labels[index]
    
    block_centers = [(b[3] + b[1])/2 for b in blocks]
    block_centers_sorted = sorted(range(len(blocks)), key=lambda k: block_centers[k])

    blocks_per_group = []
    for interval in intervals:
        add_block = []
        for block_center in block_centers_sorted:
            block = blocks[block_center]
            if interval[0] < ((block[2] + block[0]) / 2) < interval[1]: add_block.append([block_center, block])
        blocks_per_group.append(add_block)
    
    merge_tables = list()
    for blocks_group in blocks_per_group:
        for b, block in enumerate(blocks_group):
            if b+1 == len(blocks_group): break
            cid, current = block[0], block[1]
            nid, next = blocks_group[b+1][0], blocks_group[b+1][1]
            
            if block_labels[cid] == cm.TABLE_COLH.value:
                if block_labels[nid] == cm.TABLE_TCELL.value:
                    blocks.append([min(current[0], next[0]), 
                                    min(current[1], next[1]), 
                                    max(current[2], next[2]), 
                                    max(current[3], next[3])])
                    merge_tables.append(cid)
                    merge_tables.append(nid)
                    block_labels.append(cm.TABLE.value)
                    b += 1
                    continue
                
                if b+2 == len(blocks_group): break
                n_nid, n_next = blocks_group[b+2][0], blocks_group[b+2][1]
                if block_labels[nid] == cm.TABLE_SP.value and block_labels[n_nid] == cm.TABLE_TCELL.value:
                    blocks.append([min(current[0], n_next[0]), 
                                    min(current[1], n_next[1]), 
                                    max(current[2], n_next[2]), 
                                    max(current[3], n_next[3])])
                    merge_tables.append(cid)
                    merge_tables.append(nid)
                    merge_tables.append(n_nid)
                    block_labels.append(cm.TABLE.value)
                    b += 2
    
    for index in sorted(merge_tables, reverse=True):
        del blocks[index]
        del block_labels[index]
    
    for i, l in enumerate(block_labels):
        if l == cm.TABLE_TCELL.value: block_labels[i] = cm.TABLE.value
    
    id_intersections = list()
    tables = [[i, b] for i, b in enumerate(blocks) if block_labels[i] == cm.TABLE.value]
    intersections = [[] for i in range(len(tables))]
    for b, block in enumerate(blocks):
        if block_labels[b] != cm.TABLE.value:
            for t, table in enumerate(tables):
                if intersect(table[1], block):
                    intersections[t].append(block)
                    id_intersections.append(b)
    
    for i, inter in enumerate(intersections):
        if not inter: continue
        inter.append(tables[i][1])
        blocks.append([min([x[0] for x in inter]), 
                        min([x[1] for x in inter]), 
                        max([x[2] for x in inter]), 
                        max([x[3] for x in inter])])
        block_labels.append(cm.TABLE.value)
        id_intersections.append(tables[i][0])
    
    for index in sorted(set(id_intersections), reverse=True):
        del blocks[index]
        del block_labels[index]

    return blocks, block_labels, headers

def get_objects_bboxs(preds, debug = False):
    #! based on PyMuPDF, create labeled bounding box from graph model inference

    with open(preds, "rb") as f:
        node_preds = pickle.load(f)
    
    output_folder = 'output/postprocessing'
    create_folder_if_not_exists(output_folder)

    data = Papers2Graphs(path_to_config = CONFIG / 'graph' / "graphs.yaml", test=True)
    lt = LableModification(config=data.get_config())
    node_preds = lt.revert(node_preds)
    num_labels = max(node_preds) + 1

    start_index = 0

    all_files, all_blocks, all_labels, all_headers = [], [], [], []

    for idx, graph in enumerate(tqdm(data.graphs, desc='post-processing images')):

        end_index = start_index + graph.num_nodes()
        
        # Gather graph information
        page_name = data.pages[idx]['page']
        # if page_name not in ['PMC3175731_00002.pdf', 'PMC4644924_00002.pdf', 'PMC2915753_00003.pdf', 'PMC3265072_00006.pdf', 'PMC3388459_00002.pdf']:
        #     start_index = end_index
        #     continue
        bboxs = data.pages[idx]['bboxs']
        graph_node_preds = node_preds[start_index : end_index]

        # Open PDF for PyMuPDF blocks
        blocks, labels_inside_blocks_counter = list(), list()
        doc = fitz.open(PUBLAYNET_TEST / page_name)
        p = doc[0]
        elements = p.get_text("json")
        elements = json.loads(elements)
               
        for block in elements['blocks']:
            if block['type'] == 0:      # all elements
                blocks.append(block['bbox'])
                labels_inside_blocks_counter.append([0 for i in range(num_labels)])

        # Count the labels inside each box to decide the outer class (of the block)
        for bid, bbox in enumerate(bboxs):
            rectA = bbox
            for blkid, block in enumerate(blocks):
                rectB = [b/sf for b in block]
                x1 = max(min(rectA[0], rectA[2]), min(rectB[0], rectB[2]))
                y1 = max(min(rectA[1], rectA[3]), min(rectB[1], rectB[3]))
                x2 = min(max(rectA[0], rectA[2]), max(rectB[0], rectB[2]))
                y2 = min(max(rectA[1], rectA[3]), max(rectB[1], rectB[3]))
                if x1<=x2 and y1<=y2:
                    if graph_node_preds[bid] == cm.TITLE.value:
                        labels_inside_blocks_counter[blkid][graph_node_preds[bid]] += 2
                    else:
                        labels_inside_blocks_counter[blkid][graph_node_preds[bid]] += 1
                    break
        
        #! General
        block_labels = list()
        for l in labels_inside_blocks_counter:
            block_labels.append(l.index(max(l)))

        #! Tables
        new_blocks, new_labels, headers = get_tables_bbox(blocks.copy(), block_labels.copy())

        #! Images
        image_blocks = p.get_image_info()
        for img_blk in image_blocks:
            if img_blk['bbox'][3] - img_blk['bbox'][1] > 10:
                blocks.append([int(b) for b in img_blk['bbox']])  
                block_labels.append(cm.FIGURE.value)
                new_blocks.append([int(b) for b in img_blk['bbox']])  
                new_labels.append(cm.FIGURE.value)

        if debug:
            # printing blocks
            image = convert_from_path(PUBLAYNET_TEST / page_name)[0]
            image_tab = convert_from_path(PUBLAYNET_TEST / page_name)[0]
            image_graph = convert_from_path(PUBLAYNET_TEST / page_name)[0]
            draw = ImageDraw.Draw(image, 'RGBA')
            draw_tab = ImageDraw.Draw(image_tab, 'RGBA')
            draw_graph = ImageDraw.Draw(image_graph, 'RGBA')

            for b, block in enumerate(blocks):
                color = categories_colors(block_labels[b])
                draw.rectangle([b/sf for b in block], fill = color + (64,), outline=color, width=3)
            
            for b, block in enumerate(new_blocks):
                color = categories_colors(new_labels[b])
                name = str(categories_names(new_labels[b]))
                block = [b/sf for b in block]
                draw_tab.rectangle(block, fill = color + (64,), outline=color, width=3)
                
                w, h = draw.textsize(text=name, font=FONT)
                draw_tab.rectangle(block, fill = color + (64,), outline=color, width=3)
                draw_tab.rectangle((block[0], block[1], block[0] + w, block[1] - h), fill=(64, 64, 64, 255))
                draw_tab.text((block[0], block[1] - h), text=name, fill=(255, 255, 255, 255), font=FONT)
            
            for b, block in enumerate(bboxs):
                color = categories_colors(graph_node_preds[b])
                draw_graph.rectangle(block, fill = color + (64,), outline=color, width=3)
            
            page_image = page_name.split(".")[0]
            image.save(f"{output_folder}/{page_image}_orig.jpg")
            image_tab.save(f"{output_folder}/{page_image}_tab.jpg")
            image_graph.save(f"{output_folder}/{page_image}_graph.jpg")

        start_index = end_index
        new_blocks = [ [n / sf for n in block] for block in new_blocks]
        all_blocks.append(new_blocks), all_labels.append(new_labels), all_headers.append(headers), all_files.append(page_name)

    return all_files, all_blocks, all_labels, all_headers


def get_subgraph_bbox():
    #! still need to understand how to extract bbox from subgraphs

    data = Papers2Graphs(path_to_config = CONFIG / 'graph' / "graphs.yaml")
    test = ['PMC2570569_00003.pdf', 'PMC3674898_00003.pdf', 'PMC2583966_00001.pdf', 'PMC2570569_00004.pdf']

    for idx, graph in enumerate(data.graphs):
        
        page_name = data.pages[idx]['page']
        if page_name in test:
            graph.ndata['bbox'] = torch.tensor(data.pages[idx]['bboxs'])
            labels = graph.ndata['label'].tolist()
            table_ids = [i for i, l  in enumerate(labels) if (l == cm.TABLE_COLH.value or l == cm.TABLE_SP.value or l == cm.TABLE_TCELL.value)]
            tables = graph.subgraph(table_ids)
            data.get_gb().print_graph(tables, [], tables.ndata['bbox'], PUBLAYNET_TRAIN / page_name, f'{page_name.split(".")[0]}.png', tables.ndata['label'])

def write_json(all_files, all_blocks, all_labels):

    pred_bboxs = {'other': dict(), 'text': dict(), 'title': dict(), 'list': dict(), 'table': dict(), 'figure': dict(),
                'caption': dict(), 'table-colh': dict(), 'table-sp': dict(), 'table-gcell': dict(), 'table-tcell': dict(),
                'table-col': dict(), 'table-row': dict()}

    for pdf_idx, pdf_name in enumerate(tqdm(all_files, desc='writing json')):
        for block_idx, block in enumerate(all_blocks[pdf_idx]):
            cls = categories_names(all_labels[pdf_idx][block_idx]).lower()
            if pdf_name in pred_bboxs[cls].keys():
                pred_bboxs[cls][pdf_name]['bboxes'].append(block)
                pred_bboxs[cls][pdf_name]['scores'].append(1.0)
            else:
                pred_bboxs[cls][pdf_name] = dict()
                pred_bboxs[cls][pdf_name]['bboxes'] = [block]
                pred_bboxs[cls][pdf_name]['scores'] = [1.0]

    out_path = SRC / 'models/predictions/ours_bboxs.json'
    with open(out_path, 'w') as f:
            json.dump(pred_bboxs, f)
    return

if __name__ == '__main__':
    preds = OUTPUT / "all_pred/visibility-nfeatSCIBERT_BBOX-efeat-dibi-nlay4-pmodescaled-hlaydimNone"
    all_files, all_blocks, all_labels, all_headers = get_objects_bboxs(preds, debug=False)
    write_json(all_files, all_blocks, all_labels)