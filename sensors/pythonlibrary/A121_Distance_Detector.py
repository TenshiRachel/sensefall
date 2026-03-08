# /*****************************************************************************
# * | File        :	  A121_Distance_Detector.py
# * | Author      :   Waveshare team
# * | Function    :   A121 Distance Detector function
# * | Info        :
# *----------------
# * | This version:   V1.0
# * | Date        :   2025-12-18
# * | Info        :   
# ******************************************************************************
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documnetation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to  whom the Software is
# furished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS OR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#

from smbus2 import SMBus, i2c_msg
from gpiozero import Button
import time
from enum import IntFlag, IntEnum

# ===============================
# Basic registers
# ===============================
REG_VERSION                 = 0x0000
REG_PROTOCOL_STATUS          = 0x0001
REG_MEASURE_COUNTER          = 0x0002
REG_DETECTOR_STATUS          = 0x0003

# ===============================
# Distance result
# ===============================
REG_DISTANCE_RESULT          = 0x0010

# ===============================
# Peak distance
# ===============================
REG_PEAK0_DISTANCE           = 0x0011
REG_PEAK1_DISTANCE           = 0x0012
REG_PEAK2_DISTANCE           = 0x0013
REG_PEAK3_DISTANCE           = 0x0014
REG_PEAK4_DISTANCE           = 0x0015
REG_PEAK5_DISTANCE           = 0x0016
REG_PEAK6_DISTANCE           = 0x0017
REG_PEAK7_DISTANCE           = 0x0018
REG_PEAK8_DISTANCE           = 0x0019
REG_PEAK9_DISTANCE           = 0x001A

# ===============================
# Peak strength
# ===============================
REG_PEAK0_STRENGTH           = 0x001B
REG_PEAK1_STRENGTH           = 0x001C
REG_PEAK2_STRENGTH           = 0x001D
REG_PEAK3_STRENGTH           = 0x001E
REG_PEAK4_STRENGTH           = 0x001F
REG_PEAK5_STRENGTH           = 0x0020
REG_PEAK6_STRENGTH           = 0x0021
REG_PEAK7_STRENGTH           = 0x0022
REG_PEAK8_STRENGTH           = 0x0023
REG_PEAK9_STRENGTH           = 0x0024

# ===============================
# Configuration registers
# ===============================
REG_START                    = 0x0040
REG_END                      = 0x0041
REG_MAX_STEP_LENGTH          = 0x0042
REG_CLOSE_RANGE_LEAKAGE_CANCEL = 0x0043
REG_SIGNAL_QUALITY           = 0x0044
REG_MAX_PROFILE              = 0x0045
REG_THRESHOLD_METHOD         = 0x0046
REG_PEAK_SORTING             = 0x0047
REG_NUM_FRAMES_RECORDED_THRESHOLD = 0x0048
REG_FIXED_AMPLITUDE_THRESHOLD_VALUE = 0x0049
REG_THRESHOLD_SENSITIVITY    = 0x004A
REG_REFLECTOR_SHAPE          = 0x004B
REG_FIXED_STRENGTH_THRESHOLD_VALUE = 0x004C

# ===============================
# Power / control
# ===============================
REG_MEASURE_ON_WAKEUP        = 0x0080
REG_COMMAND                  = 0x0100

# ===============================
# Application
# ===============================
REG_APPLICATION_ID           = 0xFFFF

class SensorProtocolError(IntFlag):
    PROTOCOL_STATE_ERROR = 0x00000001  # Pos 0
    PACKET_LENGTH_ERROR  = 0x00000002  # Pos 1
    ADDRESS_ERROR        = 0x00000004  # Pos 2
    WRITE_FAILED         = 0x00000008  # Pos 3
    WRITE_TO_READ_ONLY   = 0x00000010  # Pos 4

class SensorProfile(IntEnum):
    PROFILE1 = 1
    PROFILE2 = 2
    PROFILE3 = 3
    PROFILE4 = 4
    PROFILE5 = 5

class SensorThresholdMethod(IntEnum):
    FIXED_AMPLITUDE = 1
    RECORDED        = 2
    CFAR            = 3
    FIXED_STRENGTH  = 4

class SensorPeakSorting(IntEnum):
    CLOSEST   = 1
    STRONGEST = 2

class SensorReflectorShape(IntEnum):
    GENERIC = 1
    PLANAR  = 2

class SensorStatus(IntFlag):
    RSS_REGISTER_OK        = (1 << 0)
    CONFIG_CREATE_OK       = (1 << 1)
    SENSOR_CREATE_OK       = (1 << 2)
    DETECTOR_CREATE_OK     = (1 << 3)
    DETECTOR_BUFFER_OK     = (1 << 4)
    SENSOR_BUFFER_OK       = (1 << 5)
    CALIBRATION_BUFFER_OK  = (1 << 6)
    CONFIG_APPLY_OK        = (1 << 7)
    SENSOR_CALIBRATE_OK    = (1 << 8)
    DETECTOR_CALIBRATE_OK  = (1 << 9)

    RSS_REGISTER_ERROR     = (1 << 16)
    CONFIG_CREATE_ERROR    = (1 << 17)
    SENSOR_CREATE_ERROR    = (1 << 18)
    DETECTOR_CREATE_ERROR  = (1 << 19)
    DETECTOR_BUFFER_ERROR  = (1 << 20)
    SENSOR_BUFFER_ERROR    = (1 << 21)
    CALIBRATION_BUFFER_ERROR = (1 << 22)
    CONFIG_APPLY_ERROR     = (1 << 23)
    SENSOR_CALIBRATE_ERROR = (1 << 24)
    DETECTOR_CALIBRATE_ERROR = (1 << 25)
    DETECTOR_ERROR         = (1 << 28)

    DETECTOR_BUSY          = (1 << 31)

ALL_ERROR = (
    SensorStatus.RSS_REGISTER_ERROR |
    SensorStatus.CONFIG_CREATE_ERROR |
    SensorStatus.SENSOR_CREATE_ERROR |
    SensorStatus.DETECTOR_CREATE_ERROR |
    SensorStatus.DETECTOR_BUFFER_ERROR |
    SensorStatus.SENSOR_BUFFER_ERROR |
    SensorStatus.CALIBRATION_BUFFER_ERROR |
    SensorStatus.CONFIG_APPLY_ERROR |
    SensorStatus.SENSOR_CALIBRATE_ERROR |
    SensorStatus.DETECTOR_CALIBRATE_ERROR |
    SensorStatus.DETECTOR_ERROR |
    SensorStatus.DETECTOR_BUSY
)

class SensorDistanceResult(IntFlag):
    NUM_DISTANCES          = 0x0F
    NEAR_START_EDGE        = (1 << 8)
    CALIBRATION_NEEDED     = (1 << 9)
    MEASURE_DISTANCE_ERROR = (1 << 10)
    TEMPERATURE            = 0xFFFF0000

class SensorCommand(IntEnum):
    APPLY_CONFIG_AND_CALIBRATE = 1
    MEASURE_DISTANCE           = 2
    APPLY_CONFIGURATION        = 3
    CALIBRATE                  = 4
    RECALIBRATE                = 5

    ENABLE_UART_LOGS            = 32
    DISABLE_UART_LOGS           = 33
    LOG_CONFIGURATION           = 34

    RESET_MODULE                = 1381192737

class SensorMode(IntEnum):
    DISTANCE_DETECTOR = 1
    PRESENCE_DETECTOR = 2
    REF_APP_BREATHING = 3
    EXAMPLE_CARGO     = 4

class A121_Distance_Detector():
    def __init__(self, bus=1, addr=0x52, BUSY_PIN=4):
        self.bus = SMBus(bus)
        self.addr = addr
        self.busy_pin = Button(BUSY_PIN)
    # ===============================
    # Write: 16-bit reg + 32-bit data
    # ===============================
    def write_u32(self, reg, data):
        buf = [
            (reg >> 8) & 0xFF,     # reg MSB
            reg & 0xFF,            # reg LSB
            (data >> 24) & 0xFF,   # data MSB
            (data >> 16) & 0xFF,
            (data >> 8) & 0xFF,
            data & 0xFF            # data LSB
        ]

        msg = i2c_msg.write(self.addr, buf)
        self.bus.i2c_rdwr(msg)

    # ===============================
    # Read: 16-bit reg → read_u32 32-bit
    # ===============================
    def read_u32(self, reg):
        # Write register address
        reg_buf = [
            (reg >> 8) & 0xFF,
            reg & 0xFF
        ]

        write_msg = i2c_msg.write(self.addr, reg_buf)
        read_msg  = i2c_msg.read(self.addr, 4)

        self.bus.i2c_rdwr(write_msg, read_msg)

        recv = list(read_msg)

        return (
            (recv[0] << 24) |
            (recv[1] << 16) |
            (recv[2] << 8)  |
            recv[3]
        )
    # ===============================
    # Set functions
    # ===============================
    def set_start_and_end_range(self, start, end):
        self.write_u32(REG_START, start)
        self.write_u32(REG_END, end)

    def set_max_step_length(self, value):
        self.write_u32(REG_MAX_STEP_LENGTH, value)

    def set_close_range_leakage_cancellation(self, value):
        self.write_u32(REG_CLOSE_RANGE_LEAKAGE_CANCEL, value)

    def set_signal_quality(self, value):
        self.write_u32(REG_SIGNAL_QUALITY, value)

    def set_max_profile(self, value):
        self.write_u32(REG_MAX_PROFILE, value)

    def set_threshold_method(self, value):
        self.write_u32(REG_THRESHOLD_METHOD, value)

    def set_peak_sorting(self, value):
        self.write_u32(REG_PEAK_SORTING, value)

    def set_num_frames_recorded_threshold(self, value):
        self.write_u32(REG_NUM_FRAMES_RECORDED_THRESHOLD, value)

    def set_fixed_amplitude_threshold_value(self, value):
        self.write_u32(REG_FIXED_AMPLITUDE_THRESHOLD_VALUE, value)

    def set_threshold_sensitivity(self, value):
        self.write_u32(REG_THRESHOLD_SENSITIVITY, value)

    def set_reflector_shape(self, value):
        self.write_u32(REG_REFLECTOR_SHAPE, value)

    def set_fixed_strength_threshold_value(self, value):
        self.write_u32(REG_FIXED_STRENGTH_THRESHOLD_VALUE, value)

    def set_measure_on_wakeup(self, value):
        self.write_u32(REG_MEASURE_ON_WAKEUP, value)

    def set_command(self, value):
        self.write_u32(REG_COMMAND, value)

    # ===============================
    # Get functions
    # ===============================
    def get_version(self):
        value = self.read_u32(REG_VERSION)
        major = (value >> 16) & 0xFF
        minor = (value >> 8) & 0xFF
        patch = value & 0xFF
        print(f"Major: 0x{major:X} Minor: 0x{minor:X} Patch: 0x{patch:X}")
        return value

    def get_protocol_status(self):
        return self.read_u32(REG_PROTOCOL_STATUS)

    def get_measure_counter(self):
        return self.read_u32(REG_MEASURE_COUNTER)

    def get_detector_status(self):
        return self.read_u32(REG_DETECTOR_STATUS)

    def get_distance_result(self):
        return self.read_u32(REG_DISTANCE_RESULT)

    def get_measure_on_wakeup(self):
        return self.read_u32(REG_MEASURE_ON_WAKEUP)

    def get_application_id(self):
        return self.read_u32(REG_APPLICATION_ID)
    
    def init(self):
        while self.busy_pin.is_pressed:
            pass

        value = self.get_application_id()
        print(f"A121_Get_Application_Id: 0x{value:X}")

        self.set_command(SensorCommand.RESET_MODULE)
        time.sleep(0.1)

        self.set_command(SensorCommand.ENABLE_UART_LOGS)

        while self.busy_pin.is_pressed:
            pass

        self.set_start_and_end_range(250, 3000)
        self.set_max_step_length(0)
        self.set_close_range_leakage_cancellation(0)
        self.set_signal_quality(15000)
        self.set_max_profile(SensorProfile.PROFILE5)
        self.set_threshold_method(SensorThresholdMethod.CFAR)
        self.set_peak_sorting(SensorPeakSorting.STRONGEST)
        self.set_num_frames_recorded_threshold(100)
        self.set_fixed_amplitude_threshold_value(100000)
        self.set_threshold_sensitivity(500)
        self.set_reflector_shape(SensorReflectorShape.GENERIC)
        self.set_fixed_strength_threshold_value(0)

        self.set_command(SensorCommand.APPLY_CONFIG_AND_CALIBRATE)

        while True:
            value = self.get_detector_status()
            print(f"A121_Get_Detector_Status: 0x{value:X}")
            if (value & ALL_ERROR) == 0:
                break
            time.sleep(0.1)

        print("A121 init OK")

    def get_distance_mm(self):
        self.set_command(SensorCommand.MEASURE_DISTANCE)

        while True:
            value = self.get_detector_status()
            if (value & ALL_ERROR) == 0:
                break
            time.sleep(0.01)

        value = self.get_distance_result()
        num = value & SensorDistanceResult.NUM_DISTANCES

        if (value & SensorDistanceResult.MEASURE_DISTANCE_ERROR) == 0:
            if (value & SensorDistanceResult.CALIBRATION_NEEDED) == 0:
                if num != 0:
                    for i in range(num):
                        distance = self.read_u32(REG_PEAK0_DISTANCE + i)
                        strength_u32 = self.read_u32(REG_PEAK0_STRENGTH + i)
                        if strength_u32 & 0x80000000:   # Determine the sign bit
                            strength = strength_u32 - 0x100000000
                        else:
                            strength = strength_u32
                        print(f"num : {i}")
                        print(f"MEASURE_DISTANCE: {distance} mm")
                        print(f"REG_PEAK0_STRENGTH: {strength / 1000.0:.2f} dB\n")
                else:
                    print("A121 NUM DISTANCES = 0")
            else:
                print("A121 CALIBRATION NEEDED")
        else:
            print("A121 MEASURE DISTANCE ERROR")


    
    