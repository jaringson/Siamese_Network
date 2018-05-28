# Siamese_Network
This project uses Residual Siamese Networks to classify data with low amount of data and training.

To run, please install ImageMagick.
```
sudo apt install imagemagick
```
Also, you will need to install the python dependencies.
```
pip install -r requirements.txt
```

Each of the files has comments and help added to parsed arguments. Please see each file for their respective useage.

To start training with any additional arguments.
```
python train_siamese.py
```
To test:
```
python test_siamese.py
```
Finally, you can sort the tested images by running:
```
python sort_tested.py
```
