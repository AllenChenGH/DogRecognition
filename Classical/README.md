To run this part:

1. create three new empty folder: _data, _pickle, _temp in the same directory with code files.
2. copy data (Images folder in Stanford Dogs Dataset) into _data file.
3. run [data_processing.py](https://github.com/AllenChenGH/DogRecognition/blob/master/Classical/data_processing.py).
4. run [feature_analysis.ipynb](https://github.com/AllenChenGH/DogRecognition/blob/master/Classical/feature_analysis.ipynb).
5. run [bag_of_visual_words.ipynb](https://github.com/AllenChenGH/DogRecognition/blob/master/Classical/bag_of_visual_words.ipynb).
6. run [classification.ipynb](https://github.com/AllenChenGH/DogRecognition/blob/master/Classical/classification.ipynb).
7. to see more results with different cluster number, change [N_CLUSTERS](https://github.com/AllenChenGH/DogRecognition/blob/master/Classical/classification.ipynb?short_path=9276ddc#L47) in [classification.ipynb](https://github.com/AllenChenGH/DogRecognition/blob/master/Classical/classification.ipynb) and run it again.

The results will be provided in classification.ipynb.
Run this part on all 120 breeds took days even a week. To reduce the running time, I reduce the breed number to 10 (randomly chosen) in my code. I ran it on all 120 breeds once to get the result and it is included in my report. If you want to test it with all breeds, you may need to change some variables in feature_analysis.ipynb and bag_of_visual_words.ipynb (from 10 to 120).

PS: you may need to change the version of opencv to correctly run [feature_analysis.ipynb](https://github.com/AllenChenGH/DogRecognition/blob/master/Classical/feature_analysis.ipynb). My Python version is 3.7 and I use opencv 3.4.2.16 on my local environment.
