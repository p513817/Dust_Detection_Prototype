#%%
import cv2
import tkinter as tk
import os, time
import PIL
from PIL import Image, ImageTk
import threading
import time
import platform as plt

# from lobe import modify_lobe
from tm2_tflite import modify_tm
from cv_dust import find_dust
from custom_io import c_io

### 影像與AI相關的參數 這些參數幾乎都在 Stream 中使用
camera_idx = 0                                                  # 相機
model_path = 'model_tm'                                         # 模型資料夾
label_idx = {0:'clean', 1:'dust', 2:'other'}                    # 編號與標籤
label_2tk = {'dust':'有粉塵', 'clean':'無粉塵', 'other':'其他'}  # 標籤與顯示內容的對應

cv_dust_idx = 1                 # 控制粉塵容錯值
cv_dust_radius_min = 3          # CV 框出粉塵的最小半徑
cv_dust_radius_max = 25         # CV 框出粉塵的最大半徑
cv_thred = 150                  # 二值化的閥值
cv_max_area = 1500              # 歸類為雜物的大小 ( 超過 )

cv_lambda = 0.5                 # OpenCV 的加權指數 
ai_lambda = 0.5                 # AI 的加權指數

### IO 控制
pin_buzzer = 16              
pin_relay = 5

io_clock = 5                    # 裝置總共會執行幾秒
t_temp = 0.5                    # 蜂鳴器的間斷時間
swith = True                    # 蜂鳴器的參數


### OS 控制
cmd_col, cmd_raw = os.get_terminal_size()

class Stream():

    def __init__(self, dev=0, model_path='model'):
        self.frame = []
        self.status = False
        self.isStop = False
        self.detectStop = False
        self.detectPause = True
        self.cap = cv2.VideoCapture(dev)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # lobe
        # self.lobe = modify_lobe(model_path)
        # self.all_confid, self.res, self.val = 0, 0, 0

        #TM
        self.tm = modify_tm(model_path)
        self.predict = None                   

    def start_stream(self):
        self.steam_thread = threading.Thread(target=self.current_frame, daemon=True, args=()).start()
    
    def start_detect(self):
        self.ai_thread = threading.Thread(target=self.detect_steam, daemon=True, args=()).start()

    def stop_stream(self):
        self.isStop = True

    def stop_detect(self):
        self.detectStop = True

    def get_frame(self):
        return (self.status, self.img_process(self.frame))

    def get_result(self):
        return (self.predict)

    def current_frame(self):
        while( not self.isStop ):
            self.status, self.frame = self.cap.read()
        self.cap.release()

    def detect_steam(self):
        while( not self.detectStop ):
            if self.status: # and not self.detectPause:
                self.predict = self.tm.detect(self.img_process(self.frame))
            else:
                self.predict = None

    def img_process(self, frame):
        if self.status:

            x0, y0 = (0, 160)
            x1, y1 = (420, 660)
            frame = frame[y0:y1, x0:x1]

            height, width, _ = frame.shape
            cut_edge = int((width-height)/2)
            new_frame = frame[:, cut_edge:width-cut_edge]

            frame = cv2.resize(new_frame, (480, 480))

            return frame

    def __del__(self):
        self.isStop = True
        self.detectStop = True

class App_Setup():

    def __init__(self):

        super().__init__()

        
        # 開啟新的視窗並進行設定
        self.window = tk.Tk()          # 開啟視窗
        self.window.title('Test')        # 視窗標題

        self.window.attributes('-fullscreen' if plt.system()=='Windows' else '-zoomed', True)  # 全螢幕 is linux '-zoomed'

        self.img_size = 480
        self.pad, self.sticky = 3, "nsew"
        self.wpad = self.pad * 2
        self.h_item = 2

        # Get screen size 
        screen_width = self.window.winfo_screenwidth() 
        screen_height = self.window.winfo_screenheight() 

        self.trg_size = min(screen_height,screen_width)-20
        self.scale_factor = (self.trg_size)/self.img_size
        
        # format for items and fonts
        self.f_size = 's'
        # self.f_blocks = {'s':25*self.scale_factor, 'm':30*self.scale_factor, 'l':40*self.scale_factor, 'xl':45*self.scale_factor}
        self.f_fonts = {'s':('Arial', int(12*self.scale_factor)),  'm':('Arial', int(16*self.scale_factor)),  'l':('Arial', int(20*self.scale_factor)), 'xl':('Arial', int(24*self.scale_factor))}

        # 設定視窗
        self.frame_cam = self.new_frame(win=self.window, row=0, col=0, rowspan=2) 
        self.frame_info = self.new_frame(win=self.window, row=0, col=1) 
        self.frame_ctrl = self.new_frame(win=self.window, row=1, col=1)

        # 相機畫面
        self.cvs = self.new_cvs(self.frame_cam, 0, 0)

        # 相機控制選項
        self.lbl_show = self.new_lbl(self.frame_info, '辨識結果', 0, 0, fg='gray')
        self.lbl_res = self.new_lbl(self.frame_info, '---', 1, 0)

        # 控制選項
        self.lbl_action = self.new_lbl(self.frame_ctrl, '選擇動作', 0, 0, fg='gray')
        self.bt_detect, self.bt_detect_text = self.new_bt(self.frame_ctrl, '開始辨識', 1, 0)
        self.bt_relay, self.bt_relay_txt = self.new_bt(self.frame_ctrl, '自動控制', 2, 0)        
        self.bt_close, self.bt_close_text = self.new_bt(self.frame_ctrl, '關閉程式', 3, 0)

        # 設定各視窗大小
        for i in range(2):
            tk.Grid.columnconfigure(self.frame_info, i, weight=1)
        for i in range(4):
            tk.Grid.rowconfigure(self.frame_ctrl, i, weight=1)

    def new_frame(self, win, row, col, columnspan=1, rowspan=1, bg_color='#d9d9d9'):
        f = tk.Frame(win, highlightbackground="gray", highlightthickness=2, bg=bg_color)
        f.grid(row=row, column=col, padx=self.pad, pady=self.pad, sticky=self.sticky, columnspan=columnspan, rowspan=rowspan)
        return f
    
    def new_cvs(self, frame, row, col):
        cvs = tk.Canvas(frame, width = self.img_size*self.scale_factor, height = self.img_size*self.scale_factor)
        cvs.grid(row=row, column=col, padx=self.pad, pady=self.pad, sticky=self.sticky)
        return cvs

    def new_lbl(self, frame, text, row, col, columnspan=1, rowspan=1, bg='#d9d9d9', fg='black', font_size=''):
        lbl = tk.Label(frame, text=text, font=self.f_fonts[self.f_size if font_size=='' else font_size], height=self.h_item)
        lbl.grid(row=row, column=col, columnspan=columnspan, rowspan=rowspan ,padx=self.pad, pady=self.pad, sticky=self.sticky)
        lbl['bg']=bg
        lbl['fg']=fg
        return lbl
    
    def new_bt(self, frame, text ,row, col, columnspan=1, rowspan=1 ,bg='#d9d9d9', fg='black'):
        str_var = tk.StringVar()
        str_var.set(text)
        bt = tk.Button(frame, textvariable=str_var , font= self.f_fonts[self.f_size], height=self.h_item, bg=bg, fg=fg)
        bt.grid(row=row, column=col, columnspan=columnspan, rowspan=rowspan, padx=self.pad, pady=self.pad, sticky=self.sticky)
        return bt, str_var

    def new_lbl_var(self, frame, text, row, col, columnspan=1, rowspan=1, bg='#d9d9d9', fg='black', font_size=''):
        str_var = tk.StringVar()
        str_var.set(text)
        lbl = tk.Label(frame, textvariable=str_var, font=self.f_fonts[self.f_size if font_size=='' else font_size], height=self.h_item)
        lbl.grid(row=row, column=col, columnspan=columnspan, rowspan=rowspan ,padx=self.pad, pady=self.pad, sticky=self.sticky)
        lbl['bg']=bg
        lbl['fg']=fg
        return lbl, str_var

    def get_screen_size(self, window):  
        return window.winfo_screenwidth(),window.winfo_screenheight() 

    def get_window_size(self, window):  
        return window.winfo_reqwidth(),window.winfo_reqheight()  

class App(App_Setup):

    def __init__(self):
        
        # 繼承 App_Setup 的 init
        # 由於 Setup 的部分太多，會影響程式閱讀，所以用繼承的方式去寫
        print('進行初始化 ...')
        super().__init__()  

        self.delay = 10
        self.isDetected = False
        self.bt_event()

        # 宣告多線程
        self.stream = Stream(camera_idx, model_path)
        
        # 影像串流與OpenCV的部份
        self.status, self.frame = 0, []
        self.mix_val = []
        self.cv_dust_img, self.cv_dust_idx = 0, 0

        # 宣告 AI 設定
        self.label_idx = label_idx
        self.label_2tk = label_2tk
        self.res = None

        # 宣告 CV 設定
        self.cv_dust_count = 1

        print('啟動多線程 ...')
        # 啟動多線程
        self.stream.start_stream()
        self.stream.start_detect()

        # IO 設定
        self.buzzer = c_io(pin_buzzer)
        self.relay = c_io(pin_relay)
        self.t_clock = io_clock
        self.t_temp = t_temp
        self.t_buzzer = 0
        self.swith = swith

        # Tkinter 
        self.update()
        self.window.mainloop()

    def update(self):
    
        # 影像串流與OpenCV運作 ，包含 CV 辨識
        self.video_stream()
            
        # AI 影像辨識 ，
        self.detect_stream()

        # 顯示按鈕資訊
        self.bt_detect_info()

        self.window.after(self.delay, self.update)


    def bt_detect_event(self):
        self.isDetected = not self.isDetected


    # 按鈕事件
    def bt_event(self):

        self.bt_detect['command'] = lambda : self.bt_detect_event()
        self.bt_close['command'] = lambda : self.close_app()

    # 影像串流加 OpenCV 
    def video_stream(self):

        # 取得影像
        self.status, self.frame = self.stream.get_frame()
        
        if self.status:
            
            # cv_dust.py 的 find_dust 
            self.cv_dust, self.cv_dust_idx, self.cv_sth_idx = find_dust(self.frame, cv_dust_radius_min, cv_dust_radius_max, cv_thred, cv_max_area)
            trg_frame = self.cv_dust if self.isDetected else self.frame
            trg_frame = cv2.resize(trg_frame, (self.trg_size, self.trg_size))

            # 放到 Canvas 上
            self.photo = ImageTk.PhotoImage(image = Image.fromarray(cv2.cvtColor(trg_frame, cv2.COLOR_BGR2RGB)))
            self.cvs.create_image(0, 0, image = self.photo, anchor="nw")         

    # AI 的部份
    def detect_stream(self):
        
        # 取得結果 0:clean , 1:dust, 2:other
        self.predict = self.stream.get_result()

        # 如果取得不到結果 predict 為 None
        if self.predict is not None:

            # 取得辨識結果
            trg = self.predict[0][0]
            self.res = label_idx[trg]

            # 方法一：AI 與 OpenCV 相比並做權重相加
            self.mix_ai_cv(ai_lambda, cv_lambda)

            # 方法二：利用 OpenCV 先進行一次除錯，再用AI辨識
            # self.aicv_results()


    def cv_results(self):

        # 當CV偵測到粉塵小於 cv_dust_count 則為乾淨的
        # 當AI偵測到其他還需要透過CV確認是否有粉塵

        if (self.cv_dust_idx < self.cv_dust_count):
            self.res = 'clean'
            self.lbl_res['text'] = self.label_2tk[self.res]
            self.lbl_res['fg'] = '#CC0000'
        else:
            self.lbl_res['text'] = self.label_2tk[self.res]
            self.lbl_res['fg'] = '#CC0000'

    def bt_detect_info(self):

        if self.isDetected and self.res is not None:
            
            self.bt_detect_text.set('停止辨識')
            self.lbl_res['text'] = self.label_2tk[self.res]
            self.lbl_res['fg'] = '#CC0000'
        else:
            self.bt_detect_text.set('開始辨識')
            self.lbl_res['text'] = '---'
            self.lbl_res['fg'] = '#d9d9d9'


    def mix_ai_cv(self, ai_lambda=0.4, cv_lambda=0.6):

        ai_lambda = ai_lambda
        cv_lambda = cv_lambda
        cv_sth_mabda = 0.9 if self.cv_sth_idx >= 1 else 0.5
        
        self.ai_val =  [0  , 0  , 0]
        self.cv_val =  [cv_lambda  , 0  , cv_sth_mabda] if self.cv_dust_idx < self.cv_dust_count else [0, cv_lambda, cv_sth_mabda]
        self.mix_val = [0  , 0  , 0  ]
        
         
        add_weight = lambda val: (val * ai_lambda)
        
        for i in range(0, len(self.predict)):
            idx = self.predict[i][0]
            val = self.predict[i][1]
            self.ai_val[idx] = val
            self.mix_val[idx] = add_weight(val) + self.cv_val[idx]

        # 將數值縮在 0~1之間
        softmax = lambda ls:[ i * 1 / sum(ls) for i in ls ]
        self.mix_val = softmax(self.mix_val)


        # 取得最大的數值
        idx = self.mix_val.index(max(self.mix_val))
        self.res = self.label_idx[idx]
        
        # 透過 OpenCv 進行二次確認
        # self.res = 'clean' if self.cv_dust_idx <= self.cv_dust_count else self.res

        self.lbl_res['text'] = self.label_2tk[self.res]
        self.lbl_res['fg'] = '#CC0000'

        # 辨識完進行IO的控制
        self.io_ctrl()

        # 顯示再終端機上
        self.terminal_log()

    def io_ctrl(self):
        
        if self.isDetected:

            if self.res is 'dust':
                self.buzzer_status = True        

                if self.buzzer.get_time() >= self.t_clock: 
                    
                    self.buzzer_status = False
                    self.buzzer.off()
                    self.relay.off()

                    self.buzzer.timer_reset()
                else:

                    self.relay.on()
                    
                    if time.time() - self.t_buzzer >= self.t_temp:
                        if self.swith:
                            self.buzzer.on()
                            self.swith = not self.swith
                        else:
                            self.buzzer.off()
                            self.swith = not self.swith

                            self.t_buzzer = time.time()

            else:
                self.buzzer_status = False
                self.buzzer.off()
                self.buzzer.timer_reset()
                self.t_buzzer = time.time()
        else:

            self.buzzer_status = False
            self.buzzer.timer_reset()
            self.io_allClose()

    def terminal_log(self):

        diver = lambda : print('='*cmd_col)


        if self.isDetected:

            os.system('clear')

            print('AUO Dust Detection', '(v1)')
            print('\n')
            print('影像辨識資訊')
            diver()
            print('{:10}\t{}'.format('辨識結果', self.res))
            print('{:10}\t{} {}'.format('粉塵數量', self.cv_dust_idx, f' (誤差 {self.cv_dust_count})'))
            print('{:10}\t{}'.format('異常物品', self.cv_sth_idx))
            print('{:10}\t{}'.format('CV分數', f'{self.cv_val[0]:.3f}, {self.cv_val[1]:.3f}, {self.cv_val[2]:.3f}'))
            print('{:10}\t{}'.format('AI分數', f'{self.ai_val[0]:.3f}, {self.ai_val[1]:.3f}, {self.ai_val[2]:.3f}'))
            print('{:10}\t{}'.format('加權分數', f'{self.mix_val[0]:.3f}, {self.mix_val[1]:.3f}, {self.mix_val[2]:.3f}'))
            print('\n')
            print('輸入輸出資訊')
            diver()
            print('{:10}\t{} ({}s)'.format('蜂鳴器', '開啟' if self.buzzer_status else '關閉', int(self.buzzer.get_time()+1) if self.buzzer_status else 0))
            print('{:10}\t{} ({}s)'.format('噴嘴', '開啟' if self.buzzer_status else '關閉', int(self.buzzer.get_time()+1) if self.buzzer_status else 0))
            
            # print(f'{time.time() - self.t_buzzer}')
        else:
            os.system('clear')

            print('AUO Dust Detection', '(v1)')
            print('\n')
            print('尚未開啟辨識 ...')

    def io_allClose(self):
        self.buzzer.off()


    def close_app(self):
        print('\n')
        self.stream.stop_detect()
        self.stream.stop_stream()
        self.window.destroy()

App()
# %%
