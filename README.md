## Satellite images classification

Classification of ships\icebergs images taken from space by Sentinel-1 satellites (Statoil/C-CORE Iceberg dataset), using C-Band radar. The problem was addressed by CNN with auxiliary inputs to the feed-forward part. Transfer knowledge techniques and different image filters for cleaning noisy data were applied as well

### Used libraries
Used python vesrion: 3.5.2

Used libraries:
* matplotlib		2.0.2
* seaborn 		0.8.1
* keras 			2.0.7
* numpy 			1.13.3
* pandas 			0.20.1
* scikit-image	0.13.1
* scikit-learn	0.19.1
* scipy			0.19.0
* h5py			2.7.1


### Data
Could be obtained here: https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data 
Both  train.json.7z and test.json.7z  should be unzipped to the project folder.


### Code structure
* CNN models: iceberg_discrimination_cnn.ipynb
* Other supervised models: iceberg_discrimination_supervised_learning.ipynb

### Models
Saved CNN models: saved_models folder

