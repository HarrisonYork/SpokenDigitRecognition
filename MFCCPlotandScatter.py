import numpy as np
import matplotlib.pyplot as plt

def read_blocks(datafile):
    blocks = []
    current_block = []

    with open(datafile, 'r') as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line == "":
                if current_block:
                    blocks.append(np.array(current_block))
                    current_block = []
            else:
                numbers = list(map(float, line.split()))
                if len(numbers) == 13:
                    current_block.append(numbers)

    if current_block:
        blocks.append(np.array(current_block))

    return blocks


file = 'Train_Arabic_Digit.txt'
all_blocks = read_blocks(file)

block_index = []
for i in range(10):
    block_index.append(660*i)

digits = []
for i in range(10):
    digits.append(all_blocks[block_index[i]])

digit = 0
block = all_blocks[block_index[digit]]
x = []
MFCCs = [[] for _ in range(13)]
MFCC1 = []
MFCC2 = []
MFCC3 = []

for b in range(660*(digit), 660*(digit+1)):
    blo = all_blocks[b]
    for frame in range(len(blo)):
        MFCC1.append(blo[frame][0])
        MFCC2.append(blo[frame][1])
        MFCC3.append(blo[frame][2])
        x.append(frame)

# for frame in range(len(block)):
#     for mfcc in range(13):
#         MFCCs[mfcc].append(block[frame][mfcc])
#     MFCC1.append(digits[digit][frame][0])
#     MFCC2.append(digits[digit][frame][1])
#     MFCC3.append(digits[digit][frame][2])


a = str(digit)

# Plot each MFCC with a unique color
# plt.figure(figsize=(7, 5))
# for i in range(13):
#     plt.plot(x, MFCCs[i], label=f'MFCC {i+1}')
#
# # Add labels, legend, and title
# plt.xlabel('Frame', size=18)
# plt.ylabel('MFCC Coefficient', size=18)
# plt.title(f'MFCCs for Digit {digit}', size=20)
# plt.legend()
# plt.show()
#
# plt.title("MFCC1 (blue), MFCC2 (red), MFCC3 (green) for Digit "+a)
# plt.xlabel("Analysis Frame")
# plt.ylabel("MFCC Coefficient")
# plt.show()

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].scatter(MFCC2, MFCC1, color='blue', alpha=0.2, s=1)
axs[0].set_title('MFCC1 vs MFCC2 for Digit ' + a, size=20)
axs[0].set_xlabel('MFCC2', size=18)
axs[0].set_ylabel('MFCC1', size=18)

axs[1].scatter(MFCC3, MFCC1, color='red', alpha=0.2, s=1)
axs[1].set_title('MFCC1 vs MFCC3 for Digit ' + a, size=20)
axs[1].set_xlabel('MFCC3', size=18)
axs[1].set_ylabel('MFCC1', size=18)

axs[2].scatter(MFCC3, MFCC2, color='green', alpha=0.2, s=1)
axs[2].set_title('MFCC2 vs MFCC3 for Digit ' + a, size=20)
axs[2].set_xlabel('MFCC3', size=18)
axs[2].set_ylabel('MFCC2', size=18)

plt.tight_layout()
plt.show()
