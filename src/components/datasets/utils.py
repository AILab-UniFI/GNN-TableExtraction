import json
import os

from types import SimpleNamespace

from src.utils.const import Categories_names
from src.utils.paths import RAW

from pdf2image import convert_from_path
from PIL import ImageDraw, ImageFont
from src.utils.const import categories_colors
from src.utils.fs import create_folder_if_not_exists

####################
#? GENERAL functions
####################

def read_json(path):
    with open(path, "r") as f:
        page_tokens = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    return page_tokens

def get_number(string):
    # names = [ "PMC4971329_00001.jpg" ]
    no_ext = string.split(".")[0]
    no_name = no_ext.split("_")[1]
    no_prefix = int(no_name)
    return no_prefix

def calculate_cell_type(cell):

    if cell["is_column_header"]:
        return Categories_names.TABLE_COLH.value, Categories_names.TABLE_COLH.name

    elif cell["is_projected_row_header"]:
        return Categories_names.TABLE_SP.value, Categories_names.TABLE_SP.name

    return Categories_names.TABLE_GCELL.value, Categories_names.TABLE_GCELL.name

def tables_to_pages(paper_tokens, pages_idxs):

    pages_dict = dict()

    # print(len(paper_tokens), end=' - ')
    for table in paper_tokens:

        table_id = table.structure_id.split("_")[-1]
        table_page = table.pdf_page_index

        if table_page not in pages_idxs:
            continue

        table_dict = pages_dict.get(table_page, {})
        table_list = table_dict.get(table_id, [])

        #! bbox, id, label,

        for cell in table.__dict__["cells"]:
            cell = cell.__dict__

            type_id, cell_type = calculate_cell_type(cell)

            if not (cell_type == 'TABLE_TCELL' and (cell["is_column_header"] or cell['is_projected_row_header'])):
                table_list.append(
                    (
                        cell["pdf_bbox"], 
                        type_id, 
                        cell_type, 
                        cell["is_column_header"], 
                        cell["row_nums"], 
                        cell["column_nums"]
                    )
                )
            if cell["pdf_text_tight_bbox"]:
                type_id, cell_type = Categories_names.TABLE_TCELL.value, Categories_names.TABLE_TCELL.name
                if not (cell_type == 'TABLE_TCELL' and (cell["is_column_header"] or cell['is_projected_row_header'])):
                    table_list.append(
                        (
                            cell["pdf_text_tight_bbox"],
                            type_id,
                            cell_type,
                            cell["is_column_header"],
                            cell["row_nums"],
                            cell["column_nums"],
                        )
                    )
        for column in table.__dict__["columns"]:
            column = column.__dict__

            type_id, column_type = Categories_names.TABLE_COL.value, Categories_names.TABLE_COL.name

            table_list.append(
                (
                    column["pdf_column_bbox"], 
                    type_id, 
                    column_type, 
                    None, 
                    None, 
                    None
                )
            )

        for row in table.__dict__["rows"]:
            row = row.__dict__

            type_id, row_type = Categories_names.TABLE_ROW.value, Categories_names.TABLE_ROW.name

            table_list.append(
                (
                    row["pdf_row_bbox"], 
                    type_id, 
                    row_type, 
                    row["is_column_header"], 
                    None, 
                    None
                )
            )

        table_dict[table_id] = table_list
        pages_dict[table_page] = table_dict

    return pages_dict

def print_annotations(annotations, phase):
    def hasTable():
        for ann in page_annotations:
            if ann[1] == Categories_names.TABLE.value:
                return True
        return False
    
    max_p = 10
    out_path = RAW / 'examples'
    create_folder_if_not_exists(out_path)
    for p, paper in enumerate(annotations["papers"].items()):
        if p == max_p: break
        for pg, page in enumerate(paper[1]['pages']):
            page_path = RAW / phase / page
            page_annotations = paper[1]['annotations'][pg]
            page_image = convert_from_path(page_path)[0]
            draw = ImageDraw.Draw(page_image, 'RGBA')
            font = ImageFont.truetype("DejaVuSans.ttf", 15)
            if not hasTable():
                for ann in page_annotations:
                    draw.rectangle((ann[0][0], ann[0][1], ann[0][2], ann[0][3]), outline = tuple(categories_colors[ann[1]]) +(255,), width=3)
                    text_size = font.getsize(ann[2])
                    draw.rectangle((ann[0][0], ann[0][1], ann[0][0] + text_size[0], ann[0][1] + text_size[1]), fill=tuple(categories_colors[ann[1]]) + (255,))
                    draw.text((ann[0][0], ann[0][1]), text=ann[2], fill=(255, 255, 255, 255),font=font)
                page_image.save(out_path / f'annotations_{p}{pg}.jpg')
            else:
                page_image_table = convert_from_path(page_path)[0]
                draw_table = ImageDraw.Draw(page_image_table, 'RGBA')
                
                page_image_cells = convert_from_path(page_path)[0]
                draw_cells = ImageDraw.Draw(page_image_cells, 'RGBA')
                
                for ann in page_annotations:
                    
                    if ann[1] in [Categories_names.TABLE_COL.value, Categories_names.TABLE_ROW.value, Categories_names.TABLE_GCELL.value]:
                        draw_table.rectangle((ann[0][0], ann[0][1], ann[0][2], ann[0][3]), outline = tuple(categories_colors[ann[1]]) +(255,), width=3)
                        text_size = font.getsize(ann[2])
                        draw_table.rectangle((ann[0][0], ann[0][1], ann[0][0] + text_size[0], ann[0][1] + text_size[1]), fill=tuple(categories_colors[ann[1]]) + (255,))
                        draw_table.text((ann[0][0], ann[0][1]), text=ann[2], fill=(255, 255, 255, 255),font=font)
                        
                    elif ann[1] in [Categories_names.TABLE_COLH.value, Categories_names.TABLE_SP.value, Categories_names.TABLE_TCELL.value]:
                        draw_cells.rectangle((ann[0][0], ann[0][1], ann[0][2], ann[0][3]), outline = tuple(categories_colors[ann[1]]) +(255,), width=3)
                        text_size = font.getsize(ann[2])
                        draw_cells.rectangle((ann[0][0], ann[0][1], ann[0][0] + text_size[0], ann[0][1] + text_size[1]), fill=tuple(categories_colors[ann[1]]) + (255,))
                        draw_cells.text((ann[0][0], ann[0][1]), text=ann[2], fill=(255, 255, 255, 255),font=font)
                        
                    else:
                        draw.rectangle((ann[0][0], ann[0][1], ann[0][2], ann[0][3]), outline = tuple(categories_colors[ann[1]]) +(255,), width=3)
                        text_size = font.getsize(ann[2])
                        draw.rectangle((ann[0][0], ann[0][1], ann[0][0] + text_size[0], ann[0][1] + text_size[1]), fill=tuple(categories_colors[ann[1]]) + (255,))
                        draw.text((ann[0][0], ann[0][1]), text=ann[2], fill=(255, 255, 255, 255),font=font)
                        
                page_image.save(out_path / f'annotations_{p}{pg}.jpg')
                page_image_table.save(out_path / f'annotations_{p}{pg}_table.jpg')
                page_image_cells.save(out_path / f'annotations_{p}{pg}_cells.jpg')
            
    return