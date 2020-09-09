#!/bin/bash

# ├── data/face_models
# │   ├── 1
# │   │   ├── saved_model.pb
# │   │   └── variables
# │   └── git_model
# │       └── saved_model.pb

sudo docker run --name tfserving -d -t --rm -p 8501:8501 \
    -v "$(pwd)/data/face_models:/models/facenet"  \
    -e MODEL_NAME=facenet \
    tensorflow/serving


# батч из одной картинки 2x2 с тремя каналами
curl -d '{
 "inputs": [
   {
     "image_batch": 
        [[[[0.8128570914268494, 0.24590837955474854, 0.06405989080667496],
           [0.42225033044815063, 0.007270897272974253, 0.7108320593833923]],
          [[0.8061221241950989, 0.87587970495224, 0.389741450548172],
           [0.4879096448421478, 0.5292569994926453, 0.5317033529281616]]]],
     "phase_train": false
   }
 ]
}' \
  -X POST http://server2.company:8501/v1/models/facenet:predict


# conda install -c pytorch faiss-cpu
