import os, sys, time

# Windows command prompt is using consolas with every pixel having:
#   width: 3px
#   height: 6px

class Display:
    WINDOW_WIDTH = 100
    WINDOW_HEIGHT = 50
    FRAME_RATE = 60
    buffer = ''
    # os.system(f'mode con: cols={WINDOW_WIDTH} lines={WINDOW_HEIGHT}')

    @staticmethod
    def printScreen(c):
        for _ in range(Display.WINDOW_HEIGHT):
            for _ in range(Display.WINDOW_WIDTH):
                Display.buffer += str(c)
        print(Display.buffer, flush=True, end='')
        Display.buffer = ''
        time.sleep(1/Display.FRAME_RATE)



while True:
    Display.printScreen("X")
    Display.printScreen("C")
    Display.printScreen("V")
    Display.printScreen("B")
    Display.printScreen("N")
    Display.printScreen("M")