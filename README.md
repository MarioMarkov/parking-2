# parking-spot-detection 🅿️🚗

Scripts to train and test a parking spot detection model

In order to run training or testing, one has to run the training main.py file with the
corresponding arguments:

### Training

```
python main.py --model_name=malex_net_pk_lot
--pk_lot_dir=pk_lot_data
--cnr_park_dir=cnr_parking_data
--dataset=both
--epochs=2
-train
```

### Testing

```
python main.py --model_name=malex_net_combined --dataset=cnr_park
```

To make predictions you have to have folder `inference/annotations` `inference/images` or set the folders in variables in the script

```
python get_predictions.py
```
