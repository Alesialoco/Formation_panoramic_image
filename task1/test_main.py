import os
from people_detected import PeopleDetector

def main():
    try:
        detector = PeopleDetector()
        detector.run()
    except SystemExit as e:
        print(e)
    except Exception as e:
        print(f'Ошибка в основной программе: {e}')


if __name__ == "__main__":
    main()
