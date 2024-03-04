import serial
import time

'''dev = serial.Serial('COM3', baudrate = 9600)

dev.write(b'1')
'''
from pyfirmata import Arduino, util

borad = Arduino('COM3')