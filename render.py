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
img = cp.array(cv2.imread("C:\All\Stuff\Projects\Images\\capybara_contrast.png", cv2.IMREAD_GRAYSCALE), dtype=cp.int32).transpose()
img = img[0:(img.shape[0] // CHUNK_WIDTH) * CHUNK_WIDTH, 0:(img.shape[1] // CHUNK_HEIGHT) * CHUNK_HEIGHT]
WINDOW_WIDTH = img.shape[0] // CHUNK_WIDTH
WINDOW_HEIGHT = img.shape[1] // CHUNK_HEIGHT


letters = cp.array(letters, dtype=cp.int32)
flats = cp.empty((WINDOW_WIDTH, WINDOW_HEIGHT, CHUNK_WIDTH * CHUNK_HEIGHT), dtype=cp.int32)

# os.system(f'mode con: cols={WINDOW_WIDTH} lines={WINDOW_HEIGHT+1}')

t2 = time.time()
for i in range(0, img.shape[0], CHUNK_WIDTH):
    for j in range(0, img.shape[1], CHUNK_HEIGHT):
        flats[i // CHUNK_WIDTH, j // CHUNK_HEIGHT] = img[i:i+CHUNK_WIDTH, j:j+CHUNK_HEIGHT].flatten()
flats = flats.reshape(-1, flats.shape[-1])
t3 = time.time()

get_diffs = cp.RawKernel(r'''
extern "C" __global__
void get_diffs(const int* x1, const int* x2, int* y, int* min_indexes) {
    
    // compute the sum of square differences of all the pixels in the block
    for (int i = 0; i < 18; i++){
        int temp = (x1[18 * blockIdx.x + i] - x2[18 * threadIdx.x + i] - 48);
        y[blockDim.x * blockIdx.x + threadIdx.x] += temp * temp;
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
                min_indexes[blockIdx.x] = i + 32;
            }
        }
    }
}
''', 'get_diffs')

y = cp.zeros((55112, 91), dtype=cp.int32)
mins = cp.zeros((55112,), dtype=cp.int32)
x2 = letters[0].transpose()
t10 = time.time()
get_diffs((55112,), (91,), (flats, letters, y, mins))

t11 = time.time()
print(mins[0:20])
print(t11 - t10)
temp2 = y

# temp0 = flats - 48
# temp1 = cp.subtract(temp0[:,:,cp.newaxis], letters)
# temp2b = cp.linalg.norm(temp1, axis=1)
# # print(temp2b[0])

temp3 = cp.argmin(temp2, axis=1)
temp4 = cp.add(temp3, INDEX_TO_UNICODE_OFFSET)
temp5 = mins.reshape(WINDOW_WIDTH, WINDOW_HEIGHT)
temp6 = temp5.transpose()
temp7 = temp6.flatten()
td = time.time()
# return chr(int(cp.argmin(diffs)) + INDEX_TO_UNICODE_OFFSET)

buffer = ''
temp7 = temp7.get()
for i in range(temp7.shape[0]):
    buffer += chr(int(temp7[i]))
# for j in range(flats.shape[1]):
#     for i in range(flats.shape[0]):
#         buffer += chr(diff2[i+j*flats.shape[0]])
#     buffer += "\n"
# # # print(flats[i,j].shape)

t4 = time.time()
print(t3-t2)
# # print(t4-t3)

# # print(td-ta)

# # print(temp0.shape)
# # print(temp1.shape)
# # print(temp2.shape)
# # print(temp3.shape)
# # print(temp4.shape)
# # print(temp5.shape)
# # print(temp6.shape)
# # print(temp7.shape)

# print(buffer)

