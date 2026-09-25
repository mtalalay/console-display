import cupy as cp, cv2, numpy as np, os, pandas as pd, time

def sort_df():
    df = pd.read_csv('C:\All\Stuff\Projects\Images\lookup_out.csv')
    df2 = df.sort_values(by='unicode')
    df2.to_csv('C:\All\Stuff\Projects\Images\lookup_out_sorted.csv', index=False)

df = pd.read_csv('C:\All\Stuff\Projects\Images\lookup_out_sorted.csv')
letters = df.values.tolist()
#trimming unicode value off
INDEX_TO_UNICODE_OFFSET = 32
letters = [letters[i][1:] for i in range(len(letters))]

for i in range(len(letters)):
    for j in range(len(letters[0])):
        # flipping white and black values
        letters[i][j] = 255 - letters[i][j]

CHUNK_HEIGHT  = 6
CHUNK_WIDTH = 3
read_img = cv2.imread("C:\All\Stuff\Projects\Images\\assets\\jett.jpg", cv2.IMREAD_GRAYSCALE)
img = read_img[0:(read_img.shape[0] // CHUNK_HEIGHT) * CHUNK_HEIGHT, 0:(read_img.shape[1] // CHUNK_WIDTH) * CHUNK_WIDTH]
WINDOW_WIDTH = img.shape[1] // CHUNK_WIDTH
WINDOW_HEIGHT = img.shape[0] // CHUNK_HEIGHT
NUM_CHUNKS = WINDOW_WIDTH * WINDOW_HEIGHT
img_cp = cp.asarray(img, dtype=cp.int32)



letters = cp.array(letters, dtype=cp.int32)
# flats = cp.empty((WINDOW_WIDTH, WINDOW_HEIGHT, CHUNK_WIDTH * CHUNK_HEIGHT), dtype=cp.int32)

os.system(f'mode con: cols={WINDOW_WIDTH} lines={WINDOW_HEIGHT+1}')

get_diffs = cp.RawKernel(r'''
extern "C" __global__
void get_diffs(const int* x1, const int* x2, int* y, char* min_indexes, const int CHUNK_WIDTH, const int CHUNK_HEIGHT, const int WINDOW_WIDTH) {
    
    // compute the sum of square differences of all the pixels in the block
    int base =  (WINDOW_WIDTH * (blockIdx.x / WINDOW_WIDTH) * CHUNK_WIDTH * CHUNK_HEIGHT) + (CHUNK_WIDTH * (blockIdx.x % WINDOW_WIDTH));
    for (int i = 0; i < CHUNK_HEIGHT; i++) {       
        for (int j = 0; j < CHUNK_WIDTH; j++) {
            int temp = (x1[base + (WINDOW_WIDTH * CHUNK_WIDTH * i) + j] - x2[(CHUNK_WIDTH * CHUNK_HEIGHT) * threadIdx.x + ((i * CHUNK_WIDTH) + j)] - 48);
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

y = cp.zeros((NUM_CHUNKS, 91), dtype=cp.int32)
mins = cp.zeros((NUM_CHUNKS,), dtype=cp.uint8)
get_diffs((NUM_CHUNKS,), (91,), (img_cp, letters, y, mins, CHUNK_WIDTH, CHUNK_HEIGHT, WINDOW_WIDTH))
buff = mins.tobytes().decode()
# print("_")
print(buff)
# print(img.shape)

