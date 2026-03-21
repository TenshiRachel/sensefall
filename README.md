## SenseFall

### What is SenseFall?

SenseFall is an edge-based elderly fall detection system, leveraging
on various sensors and AI models to detect falls.

### Setup

#### Hardware used
1. Logitech C270 Camera
2. Waveshare A121 mmWave sensor
3. Raspberry Pi 5

#### Software used
1. MoveNet Lightning UINT8 TFLite
2. YAMNet TFLite
3. Weighted Fusion
4. Python version < 3.12

#### Installation
1. Connect the hardware to your Raspberry Pi 5
2. For connection of the Waveshare A121, please refer to [this](https://www.waveshare.com/wiki/A121_Range_Sensor?srsltid=AfmBOor3pbB9m5xFgtWXXL8vwLH-XStUlC_Tz8l1dZKEz52Y-evABHBj#Hardware_Connection_2)
3. Enable I2C through Interface Options in the Raspberry Pi config and reboot
```
sudo raspi-config
sudo reboot
```
4. Move/Download the project folder on your Raspberry Pi 5
5. Use Python version < 3.12 to create a virtual environment and run the following:
```
source my_env/bin/activate
pip install -r requirements.txt

# Set GPIO pin factory
export GPIOZERO_PIN_FACTORY=lgpio
```
6. Create a .env file in the project root directory with the following:
```
nano .env
```
The env file should contain the following:
```
SMARTFALL_EMAIL_USER=smartfall.alerts@gmail.com
SMARTFALL_EMAIL_PASS=ldjw ziqa gphu aron
# Replace your_email@gmail.com to your email
SMARTFALL_EMAIL_TO=your_email@gmail.com
```
7. In your virtual environment site packages folder, navigate to the flatbuffers folder
and open compat.py
8. Change the following:
```
Line 19: import imp -> import importlib.util
Line 56: imp.find_module('numpy') -> importlib.util.find_spec('numpy')
```
9. Save the file and run main.py
```
python3 main.py
```