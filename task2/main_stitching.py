from stitching_panorama import VideoStitcher


def main():
    try:
        stitcher = VideoStitcher('config.ini')
        stitcher.run()
    except Exception as e:
        print(f'Ошибка при запуске программы: {e}')


if __name__ == '__main__':
    main()
