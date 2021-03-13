# Cracking CAPTCHAs

## Table of contents

- [Quick start](#quick-start)
- [What's included](#whats-included)
- [How to use](#how-to-use)
- [Screenshot](#screenshot)
- [Creators](#creators)
- [References](#references)

## Quick start

This tool has been tested on Ubuntu 20.04 / Intel(R) Core(TM) i5-10400F CPU @ 2.90GHz 

### Installation requirement
- Essentials:

    ```sudo apt-get install build-essential cmake unzip pkg-config```
- X-windows libraries + openGL:

    ```sudo apt-get install libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev```
- Image + video I/O libraries:

    ```sudo apt-get install libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev```
- Optimization libaries:

    ```sudo apt-get install libopenblas-dev libatlas-base-dev liblapack-dev gfortran```
- Large dataset:

    ```sudo apt-get install libhdf5-serial-dev```
- Python3-related:

    ```sudo apt-get install python3-dev python3-tk python-imaging-tk libgtk-3-dev```

#### NVIDIA

- Graphics Driver:

        ```sudo add-apt-repository ppa:graphics-drivers/ppa```
	
        ```sudo apt-get install nvidia-driver-4xx```
	
        ```sudo reboot now```
	
        Verify using the command ```nvidia-smi``` on the terminal
- CUDA:
        obtain the latest CUDA from here: ```https://developer.nvidia.com/cuda-downloads?target_os=Linux```
	
        ```chmod +x cuda_*_linux.run```
	
        ```sudo ./cuda_*.run --silent --toolkit --override```
	
        modify your shell script (e.g., `.bashrc`) to include these two paths and then ```source bashrc```:
	
            1. ```export PATH=/usr/local/cuda-xx.x/bin:$PATH```
	    
            2. ```export LD_LIBRARY_PATH=/usr/local/cuda-xx.x/lib64```
	    
        Verify using the command ```nvcc -V``` on the terminal
- cuDNN:

        obtain the latest cuDNN from here ```https://developer.nvidia.com/rdp/cudnn-download```
	
        ```tar -zxf cudnn-*.tgz```
	
        ```cd cuda```
	
        ```sudo cp -P lib64/* /usr/local/cuda/lib64/```
	
        ```sudo cp -P include/* /usr/local/cuda/include/```



#### Virtualenv
- We **highly** recommend you setup virtual environment to work to not mess up any `python` packages on your main system (you may skip this part if needed)     
     ```pip3 install virtualenv virtualenvwrapper```
    modify your shell script (e.g., `.bashrc`) to include these two paths and then ```source bashrc```:
        1. ```export WORKON_HOME=$HOME/.local/bin/.virtualenvs ```
        2. ```export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3```
        3. ```export VIRTUALENVWRAPPER_VIRTUALENV=$HOME/.local/bin/virtualenv```
        4. ```source $HOME/.local/bin/virtualenvwrapper.sh```
    afterwards, create your virtualenv using the command: ```mkvirtualenv desired_name -p python3```
    use your virtualenv using the command: ```workon desired_name```
    leave virtualenv using the command: ```deactivate```

#### TensorFlow
- Use ```tensorflow-gpu``` if your system is using NVIDIA graphics card
    ```pip install numpy```
    ```pip install tensorflow``` or ```tensorflow-gpu```
    ```pip install opencv-contrib-python```
    ```pip install scikit-image```
    ```pip install pillow```
    ```pip install imutils```
    ```pip install scikit-learn```
    ```pip install matplotlib```

- To test everything, please run the following code:
    ```$HOME: workon desired_name```
    ```$HOME: python```
        * ```import tensorflow as tf```
        * ```tf.test.is_gpu_available()```
    ```$HOME: True```

## What's included
```
.
├── really_simple_captcha/
│   ├── test_images/
│   └── trained_model/
├── wikipedia/
│   ├── data/
│   │   ├── letters/
│   │   └── training_images/
│   │── test_cases/
│   │   ├── failure_cases/
│   │   │── success_cases/
│   │   └── test_images/
│   └── trained_model/
│── dataset.py
│── helpers.py
│── lenet.py
│── test_model.py
└── train_model.py
```

## How to use

### To train the models
```usage: train_model.py [-h] -d DATASET -m TRAINED_MODEL```

### To test the models
```usage: test_model.py [-h] -i INPUT -m TRAINED_MODEL```

#### ctrl+C to leave anytime during the application

## Screenshot

<img src="images/Training.png" width="425"/> <img src="images/Trained.png" width="600"/> 
![](images/Screenshot.png)
![](images/Screenshot2.png)

## Creators
- Jae-Won Jang [https://github.com/jjang3]
- Gokce Onen [https://github.com/owlswollen/]

## References
---
#### Rosebrock, A. (2017) *Deep Learning for Computer Vision with Python: Starter Bundle*. PyImageSearch. https://books.google.com/books?id=9Ul-tgEACAAJ
#### How to break a CAPTCHA system in 15 minutes with Machine Learning. https://medium.com/@ageitgey/how-to-break-a-captcha-system-in-15-minutes-with-machine-learning-dbebb035a710
---