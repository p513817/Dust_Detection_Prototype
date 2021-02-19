import cv2
import os

def center_crop(frame):
    height, width, _ = frame.shape
    cut_edge = int((width-height)/2)
    new_frame = frame[:, cut_edge:width-cut_edge]
    
    height, width, _ = new_frame.shape
    new_size = min(height, width)
    frame = cv2.resize(frame, (new_size, new_size))
    
    return frame

if __name__ == "__main__":

    # mouse callback function
    def draw_circle(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            # cv2.circle(img,(x,y),100,(255,0,0),-1)
            print(f'{x} , {y}')

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    root = 'data/train'

    dust_path = os.path.join(root, 'dust')
    clean_path = os.path.join(root, 'clean')
    other_path = os.path.join(root, 'other')

    clean_dir = os.listdir(clean_path)
    dust_dir = os.listdir(dust_path)
    other_dir = os.listdir(other_path)

    clean_idx = len(clean_dir)
    dust_idx = len(dust_dir)
    other_idx = len(other_dir)

    cv2.namedWindow('test')
    cv2.setMouseCallback('test',draw_circle)

    crop = True
    x0, y0 = (0, 160)
    x1, y1 = (420, 660)
    
    print('\n')
    print(f'\rClean : {clean_idx} \t Dust : {dust_idx} \t Other : {other_idx}', end='')

    while(1):

        ret, frame = cap.read()
        frame = center_crop(frame[y0:y1, x0:x1])
        cv2.imshow('test', frame)

        key = cv2.waitKey(1)

        if key == ord('q'):
            print('\n', '='*50, '\n')
            break
        if key == ord('1'):
            trg_path = os.path.join(clean_path, f'{clean_idx}.jpg')
            cv2.imwrite(trg_path, frame)
            clean_idx = clean_idx+1
        if key == ord('2'):
            trg_path = os.path.join(dust_path, f'{dust_idx}.jpg')
            cv2.imwrite(trg_path, frame)
            dust_idx = dust_idx+1  
        if key == ord('3'):
            trg_path = os.path.join(other_path, f'{other_idx}.jpg')
            cv2.imwrite(trg_path, frame)
            other_idx = other_idx+1

        if key != (-1):
            print(f'\rClean : {clean_idx} \t Dust : {dust_idx} \t Other : {other_idx}', end='') 
        
    print('END')
    cap.release()
    cv2.destroyAllWindows()
    