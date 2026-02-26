## SmartFall

### What is SmartFall?

SmartFall is an edge-based elderly fall detection system, leveraging
on various sensors and AI models to detect falls.

### Setup

#### Hardware used
1. Logitech C270 Camera
2. Waveshare A121 mmWave sensor
3. CZN 15E Sound Sensor
4. Raspberry Pi 5

#### Software used
1. MoveNet Lightning UINT8
2. Weighted Fusion

#### Installation
1. Connect the hardware to your Raspberry Pi 5
2. For connection of the Waveshare A121, please refer to [this](https://www.waveshare.com/wiki/A121_Range_Sensor?srsltid=AfmBOor3pbB9m5xFgtWXXL8vwLH-XStUlC_Tz8l1dZKEz52Y-evABHBj#Hardware_Connection_2)
3. Enable I2C through Interface Options in the Raspberry Pi config
```
sudo raspi-config
```
4. Move/Download the project folder on your Raspberry Pi 5
5. Create a virtual environment and run the following:
```
pip install -r requirements.txt
```
6. In your virtual environment site packages folder, navigate to the flatbuffers folder
and open compat.py
7. Change the following:
```
Line 19: import imp -> import importlib.util
Line 56: imp.find_module('numpy') -> importlib.util.find_spec('numpy')
```
8. Save the file and run main.py
```
python3 main.py
```