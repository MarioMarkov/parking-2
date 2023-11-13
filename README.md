# parking-spot-detection
Scripts to train a parking spot detection model

Run the main file like this to start training the model:
```
python main.py --model_type=m_alex_net --model_name=malex_net_pk_lot --data_dir=pk_lot_data --epochs=2 
```
Available params for model_type : m_alex_net, alex_net, mobile_net

To make predictions you have to have folder ```inference/annotations``` ```inference/images``` or set the folders in variables in the script
```
python get_predictions.py 
```
