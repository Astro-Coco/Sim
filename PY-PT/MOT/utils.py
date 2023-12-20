from __future__ import division
import sys
import json
import time
import os
os.system("color")


class progressBar(object):
    def __init__(self, barlen=30):
        bar = ''
        for i in range(barlen):
            bar += ' '
        self.char = 0
        self.chars = ['|||','///','---','\\\\\\','|||','///','---','\\\\\\']
        print('| ' + bar + ' | '+self.chars[self.char]+' (  0.0%) in   0 s [eta:   0 s]', end='', flush=True)
        self.barlen = barlen
        self.time = time.time()
        self.start_time = self.time
        self.last_time = self.time
        self.last_eta_time = self.time
        self.progress = 0
        self.last_progress = 0
        self.speed = 0.1
        self.eta = 10

    def set_progress(self, progress):
        self.last_progress = self.progress
        self.progress = progress
        rem = ''
        for i in range(self.barlen+len(' | '+self.chars[self.char]+' (  0.0%) in   0 s [eta:   0 s]')):
            rem += '\b'
        bar = ''
        barlen = round(self.barlen*progress)
        for i in range(barlen):
            bar += '█'
        pad = ''
        for i in range(self.barlen-barlen):
            pad += ' '
        this_time = time.time()
        dt = (this_time - self.time)
        if dt > 0:
            self.speed = ((self.progress - self.last_progress)/(this_time - self.time) + 20*self.speed)/21
        if this_time - self.last_eta_time > 0.2:
            self.eta = (1-progress)/self.speed
            self.last_eta_time = self.time
        self.time = this_time
        if (self.time-self.last_time) > 0.07:
            self.last_time = self.time
            self.char += 1
            if self.char == len(self.chars):
                self.char = 0
        if progress > 0.8:
            color = '\033[32;1m'
        elif progress > 0.5:
            color = '\033[33;1m'
        else:
            color = '\033[31;1m'
        print(rem + color + self.chars[self.char].encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding) + '\033[0m' + pad + ' | ' + color + '{:5.1f}'.format(progress * 100) + '%' + '\033[0m' + ') in ' + '{:3.0f}'.format(self.time - self.start_time) + ' s [eta: ' + '{:3.0f}'.format(self.eta), end='', flush=True)


    def end(self):
        print('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b', end='', flush=True)
        print('{:5.1f}'.format(self.time-self.start_time) + ' s                  ', end='', flush=True)
        print('\n', end='', flush=True)


def read_json(filename):
    with open(filename, 'r') as file:
        data = json.loads(file.read())
    return data

def write_json(filename, data):
    with open(filename, 'w') as file:
        json.dump(data, file, sort_keys=True, indent=4)


