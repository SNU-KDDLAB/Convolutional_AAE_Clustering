from datetime import datetime
from datetime import timedelta

class Timer:
    def __init__(self, n, period_second=60): # set start point
        self.start_time = datetime.now()
        self.n = n
        self.period = 1
        self.period_second = period_second
        self.last_time = self.start_time
        self.last_i = 0
    def remaining(self, i):
        if i == 0:
            return False
        elif i > self.last_i + self.period:
            # print('i =', i, '\t last_i =', self.last_i)
            self.last_i = i
            current_time = datetime.now()
            elapsed_time = current_time - self.last_time
            self.last_time = current_time
            if elapsed_time > timedelta(seconds=self.period_second):
                print(current_time, '\t remaining time =', (current_time - self.start_time)*(self.n-i)/i)
                return True
            else:
                self.period *= 2
        return False
