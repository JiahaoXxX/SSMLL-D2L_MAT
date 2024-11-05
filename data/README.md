# Getting the Data


## VOC2012

1. Navigate to the VOC2012 data directory:
```
cd ./voc2012
```
2. Download the data:
```
curl http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar --output voc_raw.tar
```
3. Extract the data:
```
tar -xf voc_raw.tar
```
4. Format the data (If the `formatted_xxx_xxx.npy` files already exist, this step can be skipped.):
```
python format_voc2012.py
```
5. Clean up:
```
rm voc_raw.tar
```

## COCO2014

1. Navigate to the COCO2014 data directory:
```
cd ./coco2014
```
2. Download the data:
```
curl http://images.cocodataset.org/annotations/annotations_trainval2014.zip --output coco_annotations.zip
curl http://images.cocodataset.org/zips/train2014.zip --output coco_train_raw.zip
curl http://images.cocodataset.org/zips/val2014.zip --output coco_val_raw.zip
```
3. Extract the data:
```
unzip -q coco_annotations.zip
unzip -q coco_train_raw.zip
unzip -q coco_val_raw.zip
```
4. Format the data:
```
python format_coco2014.py
```
5. Clean up:
```
rm coco_annotations.zip
rm coco_train_raw.zip
rm coco_val_raw.zip
```


## NUSWIDE

*These instructions differ slightly from those for the other datasets because we re-crawled NUSWIDE.*

1. Follow the instructions [here](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html) to request a download link for the NUSWIDE images. Once approved, you will receive a link to download `Flickr.zip` which contains the images for the NUSWIDE dataset. Download this file and move it to the NUSWIDE data directory, so that the path is:
```
./nus/Flickr.zip
```
2. Navigate to the NUSWIDE data directory:
```
cd ./nus
```
3. Extract the images:
```
unzip -q Flickr.zip
```
4. Clean up:
```
rm Flickr.zip
```
5. Download the files:
```
formatted_train_labels.npy
formatted_val_labels.npy
formatted_train_images.npy
formatted_val_images.npy
```
from [here](https://drive.google.com/drive/folders/1xeQrQ5uoY1Cw6RQkipWEGJK0E2bDxkgJ?usp=drive_link) and copy them to `./nus`.


# A Uniform Format for Data

The `format_xxx.py` can be used to produce uniformly formatted image lists and labels for the framework.