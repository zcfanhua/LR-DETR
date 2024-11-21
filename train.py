import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('rtdetr-18.yaml')
    # model.load('') # loading pretrain weights
    model.train(data='datasets/data.yaml',
                cache=False,
                imgsz=640,
                epochs=150,
                batch=16,
                workers=8,
                device='0',
                # resume='', # last.pt path
                project='runs/train',
                name='exp',
                )