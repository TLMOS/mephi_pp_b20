import time
from tqdm.notebook import tqdm

class Log(object):
    def __init__(self):
        self.start = time.time()
    
    def get_time(self):
        time_passed = time.time() - self.start
        hours = int(time_passed // 3600)
        time_passed %= 3600
        minutes = int(time_passed // 60)
        time_passed %= 60
        seconds = int(time_passed)
        return '{}:{:0>2}:{:0>2}'.format(hours, minutes, seconds)
    
    def restart(self):
        self.start = time.time()
        
    def log(self, *msgs):
        for msg in msgs:
            for line in msg.split('\n'):
                print('{} {}'.format(self.get_time(), line))
                
    def tqdm(self, iterable, desc=''):
        return tqdm(iterable, desc='{} {}'.format(self.get_time(), desc))