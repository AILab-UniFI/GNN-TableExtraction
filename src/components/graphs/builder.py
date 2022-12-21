from errno import ESTALE
import json
from attrdict import AttrDict
import numpy as np

import torch
import yaml
from src.components.graphs.labels import LableModification
from src.utils.decorators import timeit

from src.utils.paths import RAW
from src.components.graphs.utils import center, distance, normalize

import fitz
from pdf2image import convert_from_path
import dgl
from PIL import ImageDraw
from src.utils.const import SCALE_FACTOR, Categories_names
import spacy

class GraphBuilder:
    """A class to build graph from pdf.
    """
    def __init__(self, config = None, path_to_config= None):

        self.config = config
        if self.config == None:
            
            with open(path_to_config) as fileobj:
                self.config = AttrDict(yaml.safe_load(fileobj))

        assert self.config != None, 'ERROR -> config file can\'t be NONE'
    
        annotations = RAW / "test.json"
        with open(annotations, 'r') as ann:
            data = json.load(ann)
            self.categories = data['categories']
      
    def print_graph(self, g, annotations, bboxs, page_path, out_path, labels=None, node_labels=True, label_converted: LableModification=None, converted=False):
        """ Print a given graph g
        
            Parameters
            ----------
            g : DGL graph
                Path to pdf page
            annotations : list()
                List of content annotations
            bboxs : list()
                Get a visibility or knn graph
            page_path : string
                If True, apply custom dataset node labels. Otherwise, it returns empty graph.
            out_path : string
                output image debug path
            labels : list()
                if None, load from graph itself
            node_labels : bool
                if True add node labels, otherwise print a generic graph (without any label)

        """
        print(f"Saving {out_path} ... ", end='')
        if labels is None and node_labels:
            labels = g.ndata['label'].tolist()
            
        if converted and label_converted != None:
            labels = list(map(
                    lambda x: label_converted.conv_to_origin[x],
                    labels
                ))

        image = convert_from_path(page_path)[0]
        start_edges, end_edges = g.edges()
        start_edges, end_edges = start_edges.tolist(), end_edges.tolist()
        draw = ImageDraw.Draw(image, 'RGBA')
        
        if node_labels:
            # printing original labels
            for orig in annotations:
                if orig[2] in ['TABLE', 'TABLE_GCELL', 'TABLE_COL', 'TABLE_ROW']:
                    continue # skip not used classes
                bbox = orig[0]
                label = int(orig[1])
                color = (self.categories[label]['color'][0], self.categories[label]['color'][1], self.categories[label]['color'][2])
                draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline= color, width=3)
                
            # printing nodes
            for b, bbox in enumerate(bboxs):
                label = int(labels[b])
                color = (self.categories[label]['color'][0], self.categories[label]['color'][1], self.categories[label]['color'][2])
                draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], fill = color + (64,), outline=color, width=3)
        
        else:
            for b, bbox in enumerate(bboxs):
                color = (127,127,127)
                draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], fill = color + (64,), outline=color, width=3)
        
        # printing endges
        for e in range(0, len(start_edges)):
            start_bb = bboxs[start_edges[e]]
            end_bb = bboxs[end_edges[e]]
            center_start = (int(start_bb[2] - (start_bb[2] - start_bb[0])/2), int(start_bb[3] - (start_bb[3] - start_bb[1])/2))
            center_end = (int(end_bb[2] - (end_bb[2] - end_bb[0])/2), int(end_bb[3] - (end_bb[3] - end_bb[1])/2))
            middle_point = [(center_start[0] + center_end[0])/2, (center_start[1] + center_end[1])/2]
            
            if [end_edges[e], start_edges[e]] in [[start_edges[o], end_edges[o]] for o in range(0, len(start_edges))]:
                draw.line((int(center_start[0]), int(center_start[1]), int(center_end[0]), int(center_end[1])), fill=(128,91,0), width=3)
            else:
                draw.line((int(center_start[0]), int(center_start[1]), int(middle_point[0]), int(middle_point[1])), fill=(0,255,0), width=3)
                draw.line((int(middle_point[0]), int(middle_point[1]), int(center_end[0]), int(center_end[1])), fill=(255,0,0), width=3)

        image.save(out_path)
        print("ok")
        return
        
    def get_graph(self, page, annotations, mode = 'visibility', set_labels=True):
        """It wraps a bunch of functions to get a DGL graph data structure from a given pdf page.
        Returns a  DGL graph g.

        Parameters
        ----------
        page : os.path
            Path to pdf page
        annotations : list()
            List of content annotations
        mode : str
            Get a visibility or knn graph
        set_labels : bool
            If True, apply custom dataset node labels. Otherwise, it returns empty graph.

        Returns
        -------
        g : a DGL graph
        
        """
        
        def get_nodes(set_labels):
            """Uses PyMuPDF to open the passed pdf pages with refactored json annotations.
            Returns lists of bboxes, contents (text), labels and lines.


            Returns
            -------
            g : DGL Graph
            
            bboxs : list()
                list of bounding box coordinates
                
            texts : list()
                list of texts
            """
            
            def get_label(rectNode):
                """Intersect PyMuPDF elements with annotations to get labels
                """
                
                label = 0 # class 'OTHER' with no matches
                centerNode = center(rectNode)
                
                for a in annotations:
                    if a[2] in ['TABLE', 'TABLE_GCELL', 'TABLE_COL', 'TABLE_ROW']:
                        continue #! skip not used classes
                    rectAnn = a[0]
                    if (centerNode[0] > rectAnn[0] and centerNode[0] < rectAnn[2] and centerNode[1] > rectAnn[1] and centerNode[1] < rectAnn[3]):
                        label = a[1]
                        if a[2] == 'FIGURE': label = -1
                        break
                        
                return label

            def get_drawings():
                draws = p.get_drawings()
                lines = []
                for path in draws:
                    for item in path["items"]:  # these are the draw commands
                        # ------------------------------------
                        # draw each entry of the 'items' list
                        # ------------------------------------
                        if item[0] == "l":  # line
                            lines.append([int(item[1].x/SCALE_FACTOR), int(item[1].y/SCALE_FACTOR), int(item[2].x/SCALE_FACTOR), int(item[2].y/SCALE_FACTOR)])
                        elif item[0] == "re":  # rectangle
                            lines.append([int(item[1].x0/SCALE_FACTOR), int((item[1].y1 - item[1].height)/SCALE_FACTOR)])
                        
                return lines

            bboxs = []
            texts = []
            labels = []
            
            doc = fitz.open(page)
            p = doc[0]
            
            ### creating node features ###
                    
            tokens = p.get_text("words")
            
            if set_labels:
                for a in annotations:
                    if a[2] == "FIGURE":
                        bboxs.append(a[0])
                        texts.append("IMAGE!")
                        labels.append(a[1])
            
            for token in tokens:
                
                bbox = [int(token[0]/SCALE_FACTOR), 
                        int(token[1]/SCALE_FACTOR),
                        int(token[2]/SCALE_FACTOR),
                        int(token[3]/SCALE_FACTOR)]
                
                if set_labels:
                    label = get_label(bbox)
                    if label == -1:
                        continue
                    labels.append(label)
                
                bboxs.append(bbox)
                texts.append(token[4])
            
            #? DRAWINGS never used, but usable
            # draws = get_drawings()
            
            assert len(bboxs) == len(texts) == len(labels)
            return bboxs, texts, labels
        
        def get_edges(mode):
            """Given a list of bounding boxes, it finds the k nearest rectangles for each one ot them.
            The 'nearness' is here conceptually treated as an 'incoming edge'.
            Returns a list of bounding box indices (u,v) - an edge - and a list of distances.

            Parameters
            ----------
            mode : str
                Get visibility or knn graph.

            Returns
            -------
            (u, v) : list()
                list of nodes indices - the edges -
            """
            
            def knn():
                """ Given a list of bounding boxes, find for each of them their k nearest ones.
                """
                
                def bound(a, ori=''):
                    if a < 0 : return 0
                    elif ori == 'h' and a > height: return height
                    elif ori == 'w' and a > width: return width
                    else: return a

                k = self.config.PREPROCESS.k
                neighbors = [] # collect list of neighbors
                window_multiplier = 2 # how much to look around bbox
                wider = (node_bbox[2] - node_bbox[0]) > (node_bbox[3] - node_bbox[1]) # if bbox wider than taller
                
                ### finding neighbors ###
                while(len(neighbors) < k and window_multiplier < 100): # keep enlarging the window until at least k bboxs are found or window too big
                    vertical_bboxs = []
                    horizontal_bboxs = []
                    neighbors = []
                    
                    if wider:
                        h_offset = int((node_bbox[2] - node_bbox[0]) * window_multiplier/4)
                        v_offset = int((node_bbox[3] - node_bbox[1]) * window_multiplier)
                    else:
                        h_offset = int((node_bbox[2] - node_bbox[0]) * window_multiplier)
                        v_offset = int((node_bbox[3] - node_bbox[1]) * window_multiplier/4)
                    
                    window = [bound(node_bbox[0] - h_offset),
                            bound(node_bbox[1] - v_offset),
                            bound(node_bbox[2] + h_offset, 'w'),
                            bound(node_bbox[3] + v_offset, 'h')] 
                    
                    [vertical_bboxs.extend(d) for d in vertical_projections[window[0]:window[2]]]
                    [horizontal_bboxs.extend(d) for d in horizontal_projections[window[1]:window[3]]]
                    
                    for v in set(vertical_bboxs):
                        for h in set(horizontal_bboxs):
                            if v == h: neighbors.append(v)
                    
                    window_multiplier += 1 # enlarge the window
                
                ### finding k nearest neighbors ###
                neighbors = list(set(neighbors))
                if node_index in neighbors:
                    neighbors.remove(node_index)
                neighbors_distances = [distance(node_bbox, bboxs[n]) for n in neighbors]
                for sd_num, sd_idx in enumerate(np.argsort(neighbors_distances)):
                    if sd_num < k:
                        if neighbors_distances[sd_idx] <= self.config.PREPROCESS.max_dist and [node_index, neighbors[sd_idx]] not in edges:
                            edges.append([neighbors[sd_idx], node_index])
                    else: break
                return
            
            def visibility():
                # two box: node and other
                # intersect if node.low <= other.high && other.low <= node.high
                
                max_dist = self.config.PREPROCESS.max_dist
                visibility_list = [[node_index, max_dist, ''], [node_index, max_dist, ''], [node_index, max_dist, ''], [node_index, max_dist, '']] # top 0, right 1, bottom 2, left 3
                node_center = [node_bbox[2]-(node_bbox[2]-node_bbox[0])/2, node_bbox[3]-(node_bbox[3]-node_bbox[1])/2]
                
                for other_index, other_bbox in enumerate(bboxs):
                    
                    if node_index == other_index: continue
                    other_center = [other_bbox[2]-(other_bbox[2]-other_bbox[0])/2, other_bbox[3]-(other_bbox[3]-other_bbox[1])/2]
                    top = other_center[1] < node_center[1]
                    right = node_center[0] < other_center[0]
                    bottom = node_center[1] < other_center[1]
                    left = other_center[0] < node_center[0]
                    vp_intersect = (node_bbox[0] <= other_bbox[2] and other_bbox[0] <= node_bbox[2]) # True if two rects "see" each other vertically, above or under
                    hp_intersect = (node_bbox[1] <= other_bbox[3] and other_bbox[1] <= node_bbox[3]) # True if two rects "see" each other horizontally, right or left
                    rect_intersect = vp_intersect and hp_intersect # True if the two rectangles are intersecting
                    
                    if rect_intersect:
                        if top:
                            visibility_list[0] = [other_index, 0]
                        elif bottom:
                            visibility_list[2] = [other_index, 0]
                            
                    elif vp_intersect:
                        if top and height/2 > visibility_list[0][1] > (node_bbox[1] - other_bbox[3]):
                            dist = (node_bbox[1] - other_bbox[3])
                            visibility_list[0] = [other_index, dist]
                        elif bottom and visibility_list[2][1] > (other_bbox[1] - node_bbox[3]):
                            dist = (other_bbox[1] - node_bbox[3])
                            visibility_list[2] = [other_index, dist]
                            
                    elif hp_intersect:
                        if right and width/2 > visibility_list[1][1] > (other_bbox[0] - node_bbox[2]):
                            dist = (other_bbox[0] - node_bbox[2])
                            visibility_list[1] = [other_index, dist]
                        elif left and visibility_list[3][1] > (node_bbox[0] - other_bbox[2]):
                            dist = (node_bbox[0] - other_bbox[2])
                            visibility_list[3] = [other_index, dist]
                            
                for pos, v in enumerate(visibility_list):
                    if node_index != v[0]:
                        
                        if pos == 0 and [node_index, v[0]] not in v_edges: # top
                            v_edges.append([v[0], node_index])
                            
                        elif pos == 3 and [node_index, v[0]] not in h_edges: # left
                            h_edges.append([v[0], node_index])
                            
                        elif pos == 2 and [v[0], node_index] not in v_edges : # bottom
                            v_edges.append([node_index, v[0]])
                            
                        elif pos == 1 and [v[0], node_index] not in h_edges: # right
                            h_edges.append([node_index, v[0]])
                
                return v_edges, h_edges

            def remove_vertical():
                    
                def points(src, dst):
                    src_center = [bboxs[src][2]-(bboxs[src][2]-bboxs[src][0])/2, bboxs[src][3]-(bboxs[src][3]-bboxs[src][1])/2]
                    dst_center = [bboxs[dst][2]-(bboxs[dst][2]-bboxs[dst][0])/2, bboxs[dst][3]-(bboxs[dst][3]-bboxs[dst][1])/2]
                    return src_center, dst_center

                def ccw(A,B,C):
                    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

                # Return true if line segments AB and CD intersect
                def intersect(A,B,C,D):
                    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
                
                indexes = []
                for idx, v in enumerate(v_edges.copy()):
                    v1, v2 = points(v[0], v[1])
                    
                    for h in h_edges:
                        h1, h2 = points(h[0], h[1])
                        if v1 != h2 and v1 != h2 and intersect(v1, v2, h1, h2):
                            indexes.append(idx)
                            break
                
                for idx in sorted(indexes, reverse=True):
                    del v_edges[idx]
                edges.extend(v_edges)
                edges.extend(h_edges)
                return
            
            edges = []
            width, height = size[0], size[1]
            
            # creating projections
            vertical_projections = [[] for i in range(width)]
            horizontal_projections = [[] for i in range(height)]
            for node_index, bbox in enumerate(bboxs):
                for hp in range(bbox[0], bbox[2]):
                    if hp >= width: hp = width - 1
                    vertical_projections[hp].append(node_index)
                for vp in range(bbox[1], bbox[3]):
                    if vp >= height: vp = height - 1
                    horizontal_projections[vp].append(node_index)
            
            if mode == 'visibility':
                # connecting nearest "visible" nodes
                v_edges = []
                h_edges = []
                for node_index, node_bbox in enumerate(bboxs):
                    v_edges, h_edges = visibility()
                remove_vertical()
            elif mode == 'knn':
                # connecting k-nearest nodes
                for node_index, node_bbox in enumerate(bboxs):
                    knn()
            else:
                raise Exception('mode should be either \'visibility\' or \'knn\'')
            
            return torch.tensor([e[0] for e in edges]), torch.tensor([e[1] for e in edges])
  
        try:
            size = convert_from_path(page)[0].size
        except:
            return None, [], []
        
        bboxs, texts, labels = get_nodes(set_labels)
        
        try:
            u, v = get_edges(mode)
        except Exception as error:
            print('Caught this error: ' + repr(error))
        
        g = dgl.graph((u, v), num_nodes=len(bboxs), idtype=torch.int32)
        g.ndata['label'] = torch.tensor(labels, dtype=torch.long)
        
        return g, bboxs, texts
    
    def set_features(self, g, bboxs, texts, size, norm = True):
        """Get a graph g as input and it sets its node and edge features.

        Args:
            g : <DGL graph>
                a pdf graph
            bboxs : list()
                pdf node coordinates
            texts : list()
                pdf textual contents
            size : tuple
                page width and height
            norm : (bool, optional) 
                If normalize features or not. Defaults to True.
        """
        
        def get_shape():
            width_bbox = bbox[2] - bbox[0]
            height_bbox = bbox[3] - bbox[1]
            center_bbox = [bbox[2] - int(width_bbox/2), bbox[3] - int(height_bbox/2)]
            return [width_bbox, height_bbox, center_bbox[0], center_bbox[1], 
                    width_bbox * height_bbox, bbox[0], bbox[1], bbox[2], bbox[3]]
        
        def get_histogram():
                """
                Function
                ----------
                Create histogram of content given a text.

                Parameters
                ----------
                text : string
                    word/s

                Returns
                ----------
                [x, y, z] - 3-dimension list with float values summing up to 1 where:
                            - x is the % of literals inside the text
                            - y is the % of numbers inside the text
                            - z is the % of other symbols i.e. @, #, .., inside the text
                """
                
                num_symbols = 0
                num_literals = 0
                num_numbers = 0
                num_others = 0
                
                histogram = [0.0000, 0.0000, 0.0000, 0.0000]

                for symbol in texts[bidx].replace(" ", ""):
                    if symbol.isalpha():
                        num_literals += 1
                    elif symbol.isdigit():
                        num_numbers += 1
                    else:
                        num_others += 1
                    num_symbols += 1

                if num_symbols != 0:
                    histogram[0] = num_literals / num_symbols
                    histogram[1] = num_numbers / num_symbols
                    histogram[2] = num_others / num_symbols
                    
                    # keep sum 1 after truncate
                    if sum(histogram) != 1.0:
                        diff = 1.0 - sum(histogram)
                        m = max(histogram) + diff
                        histogram[histogram.index(max(histogram))] = m
                
                if histogram[0:3] == [0.0,0.0,0.0]:
                    histogram[3] = 1.0
                    
                return histogram

        features = []
        maxw, maxh = 0, 0
        spacy_embedder = spacy.load("en_core_web_lg")
        
        for bidx, bbox in enumerate(bboxs):
            
            feat = [] # feature vector of node bidx
            
            # bbox features like shape, position and content
            emb_shape = get_shape()
            if emb_shape[0] > maxw: maxw = emb_shape[0]
            if emb_shape[1] > maxh: maxh = emb_shape[1]
            feat.extend(emb_shape)
            
            try:
                token_emb = spacy_embedder(texts[bidx]).vector
            except:
                token_emb = spacy_embedder("ERR0R!").vector
               
            emb_len = len(token_emb.tolist())
            feat.extend(get_histogram())
            feat.extend(token_emb.tolist())
                
            features.append(feat)
            
        u, v = g.edges()
        srcs, dsts =  u.tolist(), v.tolist()
        distances = []
        
        for i, src in enumerate(srcs):
            distances.append(distance(bboxs[src], bboxs[dsts[i]]))
        
        if norm:
            features = normalize(features, size, maxw, maxh)
            m = max(distances)
            distances = [(1 - d/m) for d in distances]
            
        g.ndata['feat'] = torch.tensor(features, dtype=torch.float32)
        g.edata['feat'] = torch.tensor(distances, dtype=torch.float32)
        
        return
    
    # todo #1 -> speed up remove_islands
    def remove_islands(self, g):
        # given a graph g, removes all the nodes too far from other classes"
        remove = []
        labels = g.ndata['label'].tolist()
        # todo -> (maybe) MUST be bidirectional for remove_islands
        graphs = dgl.to_bidirected(g)# , copy_ndata=True)
        khop = self.config.PREPROCESS.range_island
        matrix = dgl.khop_adj(graphs, khop)

        for n in range(0, graphs.num_nodes()):
            if labels[n] != Categories_names.TEXT.value: continue
            isIsland = True
            for m in range(0, graphs.num_nodes()):
                if matrix[m, n] != 0 and m != n and labels[m] != labels[n]:
                    isIsland = False
            if isIsland:
                remove.append(n)
        return torch.tensor(remove, dtype=torch.int32)

    # @timeit
    def fast_remove_islands(self, g):
        # given a graph g, removes all the nodes too far from other classes"
        labels = np.array(g.ndata['label'].tolist())
        g = dgl.to_simple(g)
        graphs = dgl.to_bidirected(g)# , copy_ndata=True)
        khop = self.config.PREPROCESS.range_island
        matrix = dgl.khop_adj(graphs, khop)

        text_idxs = np.where(labels == Categories_names.TEXT.value)[0]
        non_text_idxs = np.where(labels != Categories_names.TEXT.value)[0]
        assert len(non_text_idxs) > 0, 'ERROR - only text in graph -> what to do?'
        matrix = torch.index_select(matrix, 1, torch.from_numpy(non_text_idxs))
        matrix_sum = torch.sum(matrix, 1)
        matrix_sum = np.where(matrix_sum == 0)[0]
        # todo -> matrix_sum \ text_idxs
        return torch.from_numpy(np.intersect1d(matrix_sum,text_idxs)).type(torch.int32)
    
    def get_config(self):
        return self.config

if __name__ == "__main__":
    phases = ["train", "test"]
    num_papers = 10
    gb = GraphBuilder(path_to_config='/home/gemelli/projects/publaynet_preprocess/configs/graph/graphs.yaml')
    
    for phase in phases:
        data_path = RAW / phase
        annotations = RAW / f"{phase}.json"
        with open(annotations, 'r') as ann:
            data = json.load(ann)
            for num, paper in enumerate(data['papers'].keys()):
                if num == num_papers: break
                pages = data['papers'][paper]['pages']
                annotations = data['papers'][paper]['annotations']
                for p, page in enumerate(pages):
                    g, bboxs, texts = gb.get_graph(f"{data_path}/{page}", annotations[p])
                    gb.print_graph(g, annotations[p], bboxs, page_path=f"{data_path}/{page}", out_path=f"{RAW}/prova_{phase}_{num}{p}.jpg")