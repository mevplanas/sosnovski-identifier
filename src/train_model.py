from ultralytics import YOLO
import os 
import argparse

if __name__ == '__main__':
    # Getting the current working directory
    cwd = os.path.dirname(os.path.realpath(__file__))

    # Defining the path to the segment.yaml file 
    path_to_segment_yaml = os.path.join(cwd, '..', 'segment.yaml')

    # Parsing the training arguments 
    parser = argparse.ArgumentParser(description="Train the YOLO model")
    parser.add_argument("--data", type=str, default=path_to_segment_yaml, help="Path to the segment.yaml file")
    parser.add_argument("--weights", type=str, default="yolo11m-seg.pt", help="Path to the weights file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size") 

    # Parsing the arguments
    args = parser.parse_args()

    # Training the YOLO model
    yolo = YOLO(args.weights)
    yolo.train(data=args.data, epochs=args.epochs, batch=args.batch_size)