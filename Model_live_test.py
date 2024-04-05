from ultralytics import YOLO
import yaml



def main():
    model = YOLO('C:/Users/Jack Bowman/Documents/Programs/PytScripts/CV/YOLO_video/runs/detect/train9/weights/epoch100.pt')
    results = model.predict(source="./test_videos/china_parade.webm",show=False,save=True)
    print(results)

if __name__ == '__main__':
    main()


