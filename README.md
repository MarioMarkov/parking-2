# parking-spot-detection 🅿️🚗

Scripts to train and test a parking spot detection model

In order to run training or testing, one has to run the training main.py file with the
corresponding arguments:

### Training

```
python main.py --model_name=m_alex_net_cnr
--dataset=cnr_park
-train
--epochs=3
--model_type=m_alex
--batch_size=32
```

### Testing

```
python main.py --model_name=malex_net_combined --dataset=cnr_park
```

To make predictions you have to have folder `inference/annotations` `inference/images` or set the folders in variables in the script

```
python get_predictions.py
```
