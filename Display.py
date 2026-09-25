import cupy as cp, cv2, numpy as np, os, pandas as pd, time, sys, io
class Display:
    CHUNK_WIDTH = 3
    CHUNK_HEIGHT = 6
    NUM_CHARS = 91
    count = 0
    get_diffs = cp.RawKernel(r'''
        extern "C" __global__
        void get_diffs(const int* x1, const int* x2, int* y, char* min_indexes, const int CHUNK_WIDTH, const int CHUNK_HEIGHT, const int WINDOW_WIDTH) {
            
            // compute the sum of square differences of all the pixels in the block
            int base =  (WINDOW_WIDTH * (blockIdx.x / WINDOW_WIDTH) * CHUNK_WIDTH * CHUNK_HEIGHT) + (CHUNK_WIDTH * (blockIdx.x % WINDOW_WIDTH));
            for (int i = 0; i < CHUNK_HEIGHT; i++) {       
                for (int j = 0; j < CHUNK_WIDTH; j++) {
                    int temp = (x1[base + (WINDOW_WIDTH * CHUNK_WIDTH * i) + j] - x2[(CHUNK_WIDTH * CHUNK_HEIGHT) * threadIdx.x + ((i * CHUNK_WIDTH) + j)] - 64);
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
    
    def __init__(self, video_path, num_frames, frame_rate):
        self.vid = cv2.VideoCapture(video_path)
        if self.vid.isOpened():
            self.WIDTH  = ((int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))) // self.CHUNK_WIDTH) * self.CHUNK_WIDTH
            self.HEIGHT = ((int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))) // self.CHUNK_HEIGHT) * self.CHUNK_HEIGHT
            self.WINDOW_WIDTH = self.WIDTH // self.CHUNK_WIDTH
            self.WINDOW_HEIGHT = self.HEIGHT // self.CHUNK_HEIGHT
            self.NUM_CHUNKS = self.WINDOW_WIDTH * self.WINDOW_HEIGHT
        
        df = pd.read_csv('C:\All\Stuff\Projects\Images\lookup_out_sorted.csv')
        letters = df.values.tolist()
        #trimming unicode value off
        INDEX_TO_UNICODE_OFFSET = 32
        letters = [letters[i][1:] for i in range(len(letters))]

        for i in range(len(letters)):
            for j in range(len(letters[0])):
                # flipping white and black values
                letters[i][j] = 255 - letters[i][j]
        self.letters = cp.array(letters, dtype=cp.int32)
        self.buff = np.zeros((num_frames, self.NUM_CHUNKS), dtype=cp.uint8)
        self.FRAME_RATE = frame_rate
        self.FRAME_TIME = 1.0 / self.FRAME_RATE
        self.mins = cp.zeros((self.NUM_CHUNKS,), dtype=cp.uint8)

    def render(self, img):
        img_cp = cp.asarray(img[0:self.HEIGHT, 0:self.WIDTH], dtype=cp.int32)

        y = cp.zeros((self.NUM_CHUNKS, self.NUM_CHARS), dtype=cp.int32)
        self.get_diffs((self.NUM_CHUNKS,), (self.NUM_CHARS,), (img_cp, self.letters, y, self.mins, self.CHUNK_WIDTH, self.CHUNK_HEIGHT, self.WINDOW_WIDTH))
        self.buff[self.count] = cp.asnumpy(self.mins)
        self.count += 1

    def display(self):
        t = 0
        os.system("")
        sys.stdout.write('\033[8;' + str(self.WINDOW_HEIGHT+1) + '};' + str(self.WINDOW_WIDTH) + 't')
        sys.stdout = io.TextIOWrapper(io.BufferedWriter(sys.stdout.buffer, 1))

        for i in range(self.buff.shape[0]):
            t1 = time.time()
            # os.system( 'cls' )
            # sys.stdout.flush()
            sys.stdout.write('\r' + str(self.buff[i].tobytes().decode()))
            # sys.stdout.flush()
            t2 = time.time()
            if self.FRAME_TIME - (t2 - t1) > 0:
                time.sleep(self.FRAME_TIME - (t2 - t1))

d = Display("C:\All\Stuff\Projects\Images\\assets\\hedgehog.mp4", 900, 23.27)
os.system("")
resize = '\033[8;' + str(d.WINDOW_HEIGHT+1) + ';' + str(d.WINDOW_WIDTH) + 't'
sys.stdout.write(resize)
print("newline")
print(d.WINDOW_WIDTH)
print(d.WINDOW_HEIGHT)

# d = Display("C:\All\Stuff\Projects\Images\\hedgehog_short.mp4", 535, 30.0)
while(d.vid.isOpened()):
    ret, frame = d.vid.read()
    b = True
    if ret == True and b:
        d.render(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    else:
        break

d.display()
