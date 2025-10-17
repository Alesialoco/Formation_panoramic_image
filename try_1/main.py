import argparse
from task1 import Detection

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    detector = Detection(args.config)
    detector.process_rtsp()
    

if __name__ == "__main__":
    main()