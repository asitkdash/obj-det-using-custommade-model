# object-detection-custommade-model

I have modified all files in object detection folder so that it will work in Windows OS with out giving PYTHONPATH error.

Download TensorFlow models from below site as I couldn't upload my models folder.
https://github.com/tensorflow/models

- Collected Images and labeled them.
- run the lableimg.py from labelimg folder.
- Generated TFRecords for both training and test dataset.
    - python3 generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record
    - python3 generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record
    - export PYTHONPATH=$PYTHONPATH:pwd:pwd/slim
    - TFRecord files are updated here. Keep these files in models/research/object_detection/data folder.
- Created binary file object-detection.pbtxt and modified config file ssd_mobilenet_v1_coco.config as per our case.
   - Keep object-detection.pbtxt file in models/research/object_detection/training folder.
   - Keep ssd_mobilenet_v1_coco.config config file in models/research/object_detection/training folder.
- Trained model using binary, config and TFRecord files from models/research folder.
    - python3 object_detection/train.py --logtostderr --train_dir=object_detection/training/ --pipeline_config_path=object_detection/training/ssd_mobilenet_v1_coco.config
- Model checkpoints will be generated at models/research/object_detection/training folder
- Created the frozen-inference graph from models/research/object_detection folder using export_inference_graph.py.
    - Python3 export_inference_graph.py \ --input_type image_tensor \ --pipeline_config_path training/ssd_mobilenet_v1_coco.config \ --trained_checkpoint_prefix training/model.ckpt-6012 \ --output_directory custom_made_graph

- Modified tutorial code given by Tensorflow to take input as my frozen_inference_graph and ran the code to get the output.
- Run the object_detection_custommade.ipynb file in jupyter notebook from models/research/object_detection folder.
# Object detection Realtime
- Install opencv
- Keep object-detection-realtime python file in models/research/object_detection folder
- run object_detection_realtime.py in terminal from the above folder.
- A new window will open to recognize object.
- press "q" to kill the window.
