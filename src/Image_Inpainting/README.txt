
My Pre-trained weights are stored in '\Image_Inpainting\data\logs\weights.26-1.07.h5'
And the Vgg16 weights for Unet is stored in '\Image_Inpainting\data\logs\vgg16.h5'

Models and Convolution Layers are stored in '\Image_Inpainting\libs'
Test images and masks are stored in '\Image_Inpainting\test_for_report'

If you want to run my code and do not want to train the whole model(it cost almost 3 days for training with one RTX2080Ti),
you can go to the 'notebooks' file and check the 'total.ipynb', I put almost all the useful code in it,
if you want test my model's effect, please check the 'test.ipynb'

We use the dataset"",and you can download it from:''

My implement environment:

Keras==2.2.4
keras-tqdm==2.0.1
tensorflow==1.14

h5py
matplotlib
numpy
pandas
scipy
seaborn
tables
tqdm