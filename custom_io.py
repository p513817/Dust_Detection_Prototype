import time
import RPi.GPIO as GPIO

class c_io:
    
    def __init__(self, pin, output=True):

        self.dev = pin
        self.io = GPIO.OUT if output else GPIO.IN
        self.is_output =output
        # 初始化 GPIO
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.dev, self.io)

        # 參數
        self.t_start = time.time()      # 起始時間
        self.t_count = 0                # 計算時間 

    def timer_reset(self):
        # print('更新計時器時間')
        self.t_start = time.time()

    def get_time(self, show=False):
        # 取得 計時器時間
        self.t_count = time.time() - self.t_start
        if show: print(self.t_count)

        return self.t_count

    def on(self):
        if self.is_output:
            GPIO.output(self.dev, True)

    def off(self):
        if self.is_output:
            GPIO.output(self.dev, False)



if __name__ == "__main__":

    buzz_clock = 3
    buzz_switch = False
    buzzer = c_io(pin=5)

    buzzer.timer_reset()

    while(1):
        # 執行程式
        try:
            if buzzer.get_time() <= buzz_clock:
                if buzz_switch:
                    print('Buzzer ON', end=' ... ')
                    buzzer.on()
                    buzz_switch = not buzz_switch
                else:
                    print('Buzzer OFF', end=' ... ')            
                    buzzer.off()
                    buzz_switch = not buzz_switch

                buzzer.timer_reset()
            print(f'({int(buzzer.get_time())}s)', end='  ')
                    
        #例外中斷狀況
        except KeyboardInterrupt:
            print('\n\n')
            GPIO.cleanup()