from ultralytics import YOLO



def main():
    # Initialize the model with the pre-trained weights
    model = YOLO('yolov8n.pt')

    # Path to your dataset configuration file
    # dataset_yaml = 'C:\\Users\\Jack Bowman\\Documents\\Programs\\PytScripts\\CV\\YOLO_video\\runs\\detect\\train333\\args.yaml'
    dataset_yaml = 'C:\\Users\\Jack Bowman\\Documents\\Programs\\PytScripts\\CV\\YOLO_video\\mil_decision.yaml'

    # Train the model on your dataset
    model.train(data=dataset_yaml, epochs=400, batch=80,save_period=20)
    results = model.val(data=dataset_yaml)
    # Evaluate the model on the validation set
    print("Results on Validation Set:", results)


if __name__ == '__main__':
    main()