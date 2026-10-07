import cupy as cp, cv2, numpy as np, os, pandas as pd, time, sys, io
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_DIR / "assets"
TABLES_DIR = PROJECT_DIR / "tables"
class Display:
    CHUNK_WIDTH = 3
    CHUNK_HEIGHT = 6
    NUM_CHARS = 91
    get_diffs = cp.RawKernel(r'''
        extern "C" __global__
        void get_diffs(const int* x1, const int* x2, int* y, char* min_indexes, const int CHUNK_WIDTH, const int CHUNK_HEIGHT, const int WINDOW_WIDTH) {
            
            // compute the sum of square differences of all the pixels in the block
            int base =  (WINDOW_WIDTH * (blockIdx.x / WINDOW_WIDTH) * CHUNK_WIDTH * CHUNK_HEIGHT) + (CHUNK_WIDTH * (blockIdx.x % WINDOW_WIDTH));
            for (int i = 0; i < CHUNK_HEIGHT; i++) {       
                for (int j = 0; j < CHUNK_WIDTH; j++) {
                    int temp = (x1[base + (WINDOW_WIDTH * CHUNK_WIDTH * i) + j] - x2[(CHUNK_WIDTH * CHUNK_HEIGHT) * threadIdx.x + ((i * CHUNK_WIDTH) + j)] - 32);
                    y[blockDim.x * blockIdx.x + threadIdx.x] += temp * temp; 
                }
            }
            
            // take the square root and prevent 0 for computed value by adding 1 (change later?)
            y[blockDim.x * blockIdx.x + threadIdx.x] = (int) sqrt((float) y[blockDim.x * blockIdx.x + threadIdx.x]) + 1;

            // for one of the threads wait for all threads to complete, then find minimum.
            if (threadIdx.x == 0) {
                int done = 1;
                do {
                    for (int i = 0; i < 91; i++){
                        done &= y[blockDim.x * blockIdx.x + i];
                    }
                } while (!done);
                int min = 880;
                for (int i = 0; i < 91; i++){
                    if (y[blockDim.x * blockIdx.x + i] < min) {
                        min = y[blockDim.x * blockIdx.x + i];
                        min_indexes[blockIdx.x] = (char) (i + 32);
                    }
                }
            }
        }
        ''', 'get_diffs')
    
    def __init__(self, video_path):
        self.vid = cv2.VideoCapture(video_path)
        if not self.vid.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        self.WIDTH = (
            int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)) // self.CHUNK_WIDTH
        ) * self.CHUNK_WIDTH
        self.HEIGHT = (
            int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)) // self.CHUNK_HEIGHT
        ) * self.CHUNK_HEIGHT
        self.WINDOW_WIDTH = self.WIDTH // self.CHUNK_WIDTH
        self.WINDOW_HEIGHT = self.HEIGHT // self.CHUNK_HEIGHT
        self.NUM_CHUNKS = self.WINDOW_WIDTH * self.WINDOW_HEIGHT

        self.NUM_FRAMES = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.FRAME_RATE = float(self.vid.get(cv2.CAP_PROP_FPS))
        if self.NUM_FRAMES <= 0:
            raise ValueError("The video metadata does not provide a valid frame count.")
        if self.FRAME_RATE <= 0:
            raise ValueError("The video metadata does not provide a valid frame rate.")
        self.count = 0
        
        df = pd.read_csv(TABLES_DIR / "lookup_3x6.csv")
        letters = df.values.tolist()
        #trimming unicode value off
        INDEX_TO_UNICODE_OFFSET = 32
        letters = [letters[i][1:] for i in range(len(letters))]

        self.letters = cp.array(letters, dtype=cp.int32)
        self.buff = np.zeros((self.NUM_FRAMES, self.NUM_CHUNKS), dtype=np.uint8)
        self.FRAME_TIME = 1.0 / self.FRAME_RATE
        self.mins = cp.zeros((self.NUM_CHUNKS,), dtype=cp.uint8)

    def render(self, img):
        img_cp = cp.asarray(img[0:self.HEIGHT, 0:self.WIDTH], dtype=cp.int32)

        y = cp.zeros((self.NUM_CHUNKS, self.NUM_CHARS), dtype=cp.int32)
        self.get_diffs((self.NUM_CHUNKS,), (self.NUM_CHARS,), (img_cp, self.letters, y, self.mins, self.CHUNK_WIDTH, self.CHUNK_HEIGHT, self.WINDOW_WIDTH))
        if self.count >= len(self.buff):
            extra_frames = max(1, len(self.buff))
            self.buff = np.concatenate((
                self.buff,
                np.zeros((extra_frames, self.NUM_CHUNKS), dtype=np.uint8),
            ))
        self.buff[self.count] = cp.asnumpy(self.mins)
        self.count += 1

    def display(self):
        os.system("")
        sys.stdout.write(
            f"\033[8;{self.WINDOW_HEIGHT + 1};{self.WINDOW_WIDTH}t"
        )
        sys.stdout.flush()

        sys.stdout.write('\033[2J')
        sys.stdout.flush()

        for i in range(self.count):
            t1 = time.time()
            frame = self.buff[i].tobytes().decode()
            rows = [
                frame[row * self.WINDOW_WIDTH:(row + 1) * self.WINDOW_WIDTH]
                for row in range(self.WINDOW_HEIGHT)
            ]
            output = ''.join(
                f'\033[{row_index + 1};1H{row_text}'
                for row_index, row_text in enumerate(rows)
            )
            sys.stdout.write(output)
            sys.stdout.flush()
            t2 = time.time()
            if self.FRAME_TIME - (t2 - t1) > 0:
                time.sleep(self.FRAME_TIME - (t2 - t1))

d = Display(str(ASSETS_DIR / "blinding_lights.mp4"))
while(d.vid.isOpened()):
    ret, frame = d.vid.read()
    if ret:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.convertScaleAbs(gray_frame, alpha=1.3, beta=0)
        d.render(gray_frame)
    else:
        break

d.display()
