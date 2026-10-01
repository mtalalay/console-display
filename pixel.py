import cv2, os, numpy, pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
LETTERS_DIR = PROJECT_DIR / "letters"
BLOCK_WIDTH = 22
BLOCK_HEIGHT = 20

df = pd.read_csv(PROJECT_DIR / "lookup.csv")
for file in os.listdir(LETTERS_DIR):
    print(file)
    img = cv2.imread(str(LETTERS_DIR / file), cv2.IMREAD_GRAYSCALE)

    def check_block(col, row):
        color = 0
        for c in range(col, col + BLOCK_WIDTH):
            for r in range(row, row + BLOCK_HEIGHT):
                color += img[r][c]
        return round(color / ( BLOCK_WIDTH * BLOCK_HEIGHT))

    row = [file[:-4]]
    for br in range(0, len(img), BLOCK_HEIGHT):
        for bc in range(0, len(img[0]), BLOCK_WIDTH):
            row.append(check_block(bc, br))
    df.loc[len(df)] = row
df.to_csv(PROJECT_DIR / "lookup_out.csv", index=False)

