# rockPaperScissorAI

Dataset can be downloaded with the following link:
https://drive.google.com/open?id=1tw9W_OWsSr7uEsCCj4sLu0urAfpY4Odq

github repository link:
https://github.com/lhphanUCI/CS175_RPS_AI

AI that will detect playable move and classify move <br/><br/>

augment.py - Video augmentation functions combined into one module. <br/>
augmentation.py - Return new augmented X by brightness or rotation. Accomplished w/ opencv. <br/>
datasetCollector.py - Code that will activate webcam to collect dataset <br/>
imageContrast.py - Return new augmented X by modifying contrast w/ tensorflow <br/>
imageCrop.py - Return new augmented X by cropping w/ tensorflow <br/>
imageRotate.py - Return new augmented X by doing rotation w/ tensorflow. <br/>
loadDataset.py - Loads directories and csv files to return (X, Y) <br/>
main.py - Runs webcam to classify moves in real time <br/>
models.py - Module containing Tensorflow model definitions. <br/>
models-train.ipynb - Jupyter Notebook used to train and cross-validate models. <br/>
project.ipynb - Jupyter Notebook to demonstrate project<br/>
settings.json - Contains various values that will be used globally among all files <br/>
settings.py - Return values set in settings.json file <br/>
window_utils.py - Misc. custom functions used by TKinter application, including framerate limiter. <br/>
