# Personal Items Detection
## Background
There are many items on my table, so an AI item dectector would definitely help me to find and take my items before leaving home. Therefore, in this project, the model is trained to detect my cell phone, watch, calculator and mask and mark them with squares of different colors. In addition, it can also check if I've worn a mask and my watch. It enables me to save much time when getting prepared to commute on every busy weekday.

# Procedure
## Hardware
* Jetson Nano 2GB Development Kit
* Webcam (V4L2 camera)

## Software
* JetPack 4.6 in Docker container
* PyTorch
* TensorRT

## Preparation
Open the following directory and then create a new folder to store your data:
```bash
$ cd ~/jetson-inference/python/training/detection/ssd/data
$ mkdir <PROJECT NAME>
```
In this new directory, you are going to create labels for your objects in a text file.
* Create a text file and name it "labels.txt".
* Each line contains one label.
![image](https://github.com/jypipi/Jetson-AI-Certificate-Project/blob/main/Images/txt.png)

## Launch a Container
```bash
$ git clone --recursive https://github.com/dusty-nv/jetson-inference # Only run this line for the first time
$ cd ~/jetson-inference
$ docker/run.sh
```

## Data Collection
This project utilizes the webcam to collect my own data by capturing images and pointing out the detection targets from various positions, orientations and backgrounds.

* Test the Webcam
```bash
$ video-viewer /dev/video0
```

* Image Capture
```bash
$ cd ~/jetson-inference/python/training/detection/ssd/
$ camera-capture /dev/video0
```
In the pop-up window, add the dataset path of the label.txt before collecting data.
For my dataset, I collected more than 800 images in total.

## Train the Model
Once data collection is completed, run the `train_model.sh` to start training:
```bash
$ cd ~/jetson-inference/python/training/detection/ssd/
$ sh train_model.sh
```
Due to the limited memory of the Jetson nano 2GB kit, I used `--batch-size=2 --workers=1` to keep it from getting frozen.

When training is done, convert the model from PyTorch to ONNX so that it can be loaded with TensorRT:
```bash
$ python3 onnx_export.py --model-dir=models/<MODEL NAME>
```

## Run the Model
Now run the model to see how it works:
```bash
$ cd ~/jetson-inference/python/training/detection/ssd/
$ sh run_model.sh
```
When it detects an item, the object would be highlighted with a colored rectangle, with its name and the detection belief shown at the top.
![image](https://github.com/jypipi/Jetson-AI-Certificate-Project/blob/main/Images/Result.jpg)

# Issues and Troubleshooting
## Fail to Open Webcam when ssh to Jetson Nano with Windows
This issue is common and difficult to fix if using Windows system. One of the easier ways to solve/eliminate it is to run `xhost +` before ssh to the Jetson nano board with a virtual machine running in Linux system, or connect the kit to a monitor and work on the project in it directly.

## Error with Converting to ONNX with Custom Dataset
`onnx IsADirectoryError: [Errno 21] Is a directory`
When training the model, it is recommended to take a look at the validation loss of each epoch. If the dataset is bad, this loss may be nan, which would cause a bad model and failure to complete the conversion.
