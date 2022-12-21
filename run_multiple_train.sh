#!/bin/sh

#!########
#! knn
#!########

#? knn - fixed - 3 - 1000
python src/models/model_train.py --mode=knn --features BBOX --edge_features=False --bidirectional=False --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=knn --features BBOX --edge_features=False --bidirectional=True --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=knn --features BBOX --edge_features=True --bidirectional=False --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=knn --features BBOX --edge_features=True --bidirectional=True --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=knn --features BBOX REPR --edge_features=False --bidirectional=False --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=knn --features BBOX REPR --edge_features=False --bidirectional=True --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=knn --features BBOX REPR --edge_features=True --bidirectional=False --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=knn --features BBOX REPR --edge_features=True --bidirectional=True --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=knn --features BBOX SPACY --edge_features=False --bidirectional=False --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=knn --features BBOX SPACY --edge_features=False --bidirectional=True --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=knn --features BBOX SPACY --edge_features=True --bidirectional=False --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=knn --features BBOX SPACY --edge_features=True --bidirectional=True --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=knn --features BBOX SCIBERT --edge_features=False --bidirectional=False --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=knn --features BBOX SCIBERT --edge_features=False --bidirectional=True --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=knn --features BBOX SCIBERT --edge_features=True --bidirectional=False --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=knn --features BBOX SCIBERT --edge_features=True --bidirectional=True --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=knn --features BBOX REPR SPACY --edge_features=False --bidirectional=False --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=knn --features BBOX REPR SPACY --edge_features=False --bidirectional=True --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=knn --features BBOX REPR SPACY --edge_features=True --bidirectional=False --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=knn --features BBOX REPR SPACY --edge_features=True --bidirectional=True --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=knn --features BBOX REPR SCIBERT --edge_features=False --bidirectional=False --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=knn --features BBOX REPR SCIBERT --edge_features=False --bidirectional=True --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=knn --features BBOX REPR SCIBERT --edge_features=True --bidirectional=False --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=knn --features BBOX REPR SCIBERT --edge_features=True --bidirectional=True --n_layers=3 --mode_params=fixed --h_layer_dim=1000

#? knn - scaled - 3 - 100000 (tot)
python src/models/model_train.py --mode=knn --features BBOX --edge_features=False --bidirectional=False --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=knn --features BBOX --edge_features=False --bidirectional=True --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=knn --features BBOX --edge_features=True --bidirectional=False --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=knn --features BBOX --edge_features=True --bidirectional=True --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=knn --features BBOX REPR --edge_features=False --bidirectional=False --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=knn --features BBOX REPR --edge_features=False --bidirectional=True --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=knn --features BBOX REPR --edge_features=True --bidirectional=False --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=knn --features BBOX REPR --edge_features=True --bidirectional=True --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=knn --features BBOX SPACY --edge_features=False --bidirectional=False --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=knn --features BBOX SPACY --edge_features=False --bidirectional=True --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=knn --features BBOX SPACY --edge_features=True --bidirectional=False --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=knn --features BBOX SPACY --edge_features=True --bidirectional=True --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=knn --features BBOX SCIBERT --edge_features=False --bidirectional=False --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=knn --features BBOX SCIBERT --edge_features=False --bidirectional=True --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=knn --features BBOX SCIBERT --edge_features=True --bidirectional=False --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=knn --features BBOX SCIBERT --edge_features=True --bidirectional=True --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=knn --features BBOX REPR SPACY --edge_features=False --bidirectional=False --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=knn --features BBOX REPR SPACY --edge_features=False --bidirectional=True --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=knn --features BBOX REPR SPACY --edge_features=True --bidirectional=False --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=knn --features BBOX REPR SPACY --edge_features=True --bidirectional=True --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=knn --features BBOX REPR SCIBERT --edge_features=False --bidirectional=False --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=knn --features BBOX REPR SCIBERT --edge_features=False --bidirectional=True --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=knn --features BBOX REPR SCIBERT --edge_features=True --bidirectional=False --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=knn --features BBOX REPR SCIBERT --edge_features=True --bidirectional=True --n_layers=3 --mode_params=scaled --params_no=100000

#!########
#! visibility
#!########

#? visibility - fixed - 3 - 1000
python src/models/model_train.py --mode=visibility --features BBOX --edge_features=False --bidirectional=False --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=visibility --features BBOX --edge_features=False --bidirectional=True --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=visibility --features BBOX --edge_features=True --bidirectional=False --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=visibility --features BBOX --edge_features=True --bidirectional=True --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=visibility --features BBOX REPR --edge_features=False --bidirectional=False --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=visibility --features BBOX REPR --edge_features=False --bidirectional=True --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=visibility --features BBOX REPR --edge_features=True --bidirectional=False --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=visibility --features BBOX REPR --edge_features=True --bidirectional=True --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=visibility --features BBOX SPACY --edge_features=False --bidirectional=False --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=visibility --features BBOX SPACY --edge_features=False --bidirectional=True --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=visibility --features BBOX SPACY --edge_features=True --bidirectional=False --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=visibility --features BBOX SPACY --edge_features=True --bidirectional=True --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=visibility --features BBOX SCIBERT --edge_features=False --bidirectional=False --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=visibility --features BBOX SCIBERT --edge_features=False --bidirectional=True --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=visibility --features BBOX SCIBERT --edge_features=True --bidirectional=False --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=visibility --features BBOX SCIBERT --edge_features=True --bidirectional=True --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=visibility --features BBOX REPR SPACY --edge_features=False --bidirectional=False --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=visibility --features BBOX REPR SPACY --edge_features=False --bidirectional=True --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=visibility --features BBOX REPR SPACY --edge_features=True --bidirectional=False --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=visibility --features BBOX REPR SPACY --edge_features=True --bidirectional=True --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=visibility --features BBOX REPR SCIBERT --edge_features=False --bidirectional=False --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=visibility --features BBOX REPR SCIBERT --edge_features=False --bidirectional=True --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=visibility --features BBOX REPR SCIBERT --edge_features=True --bidirectional=False --n_layers=3 --mode_params=fixed --h_layer_dim=1000
python src/models/model_train.py --mode=visibility --features BBOX REPR SCIBERT --edge_features=True --bidirectional=True --n_layers=3 --mode_params=fixed --h_layer_dim=1000

#? visibility - scaled - 3 - 100000 (tot)
python src/models/model_train.py --mode=visibility --features BBOX --edge_features=False --bidirectional=False --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=visibility --features BBOX --edge_features=False --bidirectional=True --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=visibility --features BBOX --edge_features=True --bidirectional=False --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=visibility --features BBOX --edge_features=True --bidirectional=True --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=visibility --features BBOX REPR --edge_features=False --bidirectional=False --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=visibility --features BBOX REPR --edge_features=False --bidirectional=True --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=visibility --features BBOX REPR --edge_features=True --bidirectional=False --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=visibility --features BBOX REPR --edge_features=True --bidirectional=True --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=visibility --features BBOX SPACY --edge_features=False --bidirectional=False --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=visibility --features BBOX SPACY --edge_features=False --bidirectional=True --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=visibility --features BBOX SPACY --edge_features=True --bidirectional=False --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=visibility --features BBOX SPACY --edge_features=True --bidirectional=True --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=visibility --features BBOX SCIBERT --edge_features=False --bidirectional=False --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=visibility --features BBOX SCIBERT --edge_features=False --bidirectional=True --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=visibility --features BBOX SCIBERT --edge_features=True --bidirectional=False --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=visibility --features BBOX SCIBERT --edge_features=True --bidirectional=True --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=visibility --features BBOX REPR SPACY --edge_features=False --bidirectional=False --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=visibility --features BBOX REPR SPACY --edge_features=False --bidirectional=True --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=visibility --features BBOX REPR SPACY --edge_features=True --bidirectional=False --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=visibility --features BBOX REPR SPACY --edge_features=True --bidirectional=True --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=visibility --features BBOX REPR SCIBERT --edge_features=False --bidirectional=False --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=visibility --features BBOX REPR SCIBERT --edge_features=False --bidirectional=True --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=visibility --features BBOX REPR SCIBERT --edge_features=True --bidirectional=False --n_layers=3 --mode_params=scaled --params_no=100000
python src/models/model_train.py --mode=visibility --features BBOX REPR SCIBERT --edge_features=True --bidirectional=True --n_layers=3 --mode_params=scaled --params_no=100000
