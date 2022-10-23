import numpy as np
import sys
import scipy.io as io
import random

# ball_18
ball_18_0 = io.loadmat("D:\实验\cwru\CaseWesternReserveUniversityData-master\\12k Drive End Bearing Fault Data\\118")["X118_DE_time"].tolist()
# ball_18_1 = io.loadmat("./CWRU/ball/ball18/119")["X119_DE_time"].tolist()
# ball_18_2 = io.loadmat("./CWRU/ball/ball18/120")["X120_DE_time"].tolist()
# ball_18_3 = io.loadmat("./CWRU/ball/ball18/121")["X121_DE_time"].tolist()
ball_18 = [ball_18_0]
# ball_18 = [ball_18_0, ball_18_1, ball_18_2, ball_18_3]

# ball_36
ball_36_0 = io.loadmat("D:\实验\cwru\CaseWesternReserveUniversityData-master\\12k Drive End Bearing Fault Data\\185")["X185_DE_time"].tolist()
# ball_36_1 = io.loadmat("./CWRU/ball/ball36/186")["X186_DE_time"].tolist()
# ball_36_2 = io.loadmat("./CWRU/ball/ball36/187")["X187_DE_time"].tolist()
# ball_36_3 = io.loadmat("./CWRU/ball/ball36/188")["X188_DE_time"].tolist()
ball_36 = [ball_36_0]
# ball_36 = [ball_36_0, ball_36_1, ball_36_2, ball_36_3]

# ball_54
ball_54_0 = io.loadmat("D:\实验\cwru\CaseWesternReserveUniversityData-master\\12k Drive End Bearing Fault Data\\222")["X222_DE_time"].tolist()
# ball_54_1 = io.loadmat("./CWRU/ball/ball54/223")["X223_DE_time"].tolist()
# ball_54_2 = io.loadmat("./CWRU/ball/ball54/224")["X224_DE_time"].tolist()
# ball_54_3 = io.loadmat("./CWRU/ball/ball54/225")["X225_DE_time"].tolist()
ball_54 = [ball_54_0]
# ball_54 = [ball_54_0, ball_54_1, ball_54_2, ball_54_3]

# inner_18
inner_18_0 = io.loadmat("D:\实验\cwru\CaseWesternReserveUniversityData-master\\12k Drive End Bearing Fault Data\\105")["X105_DE_time"].tolist()
# inner_18_1 = io.loadmat("./CWRU/inner/inner18/106")["X106_DE_time"].tolist()
# inner_18_2 = io.loadmat("./CWRU/inner/inner18/107")["X107_DE_time"].tolist()
# inner_18_3 = io.loadmat("./CWRU/inner/inner18/108")["X108_DE_time"].tolist()
inner_18 = [inner_18_0]
# inner_18 = [inner_18_0, inner_18_1, inner_18_2, inner_18_3]

# inner_36
inner_36_0 = io.loadmat("D:\实验\cwru\CaseWesternReserveUniversityData-master\\12k Drive End Bearing Fault Data\\169")["X169_DE_time"].tolist()
# inner_36_1 = io.loadmat("./CWRU/inner/inner36/170")["X170_DE_time"].tolist()
# inner_36_2 = io.loadmat("./CWRU/inner/inner36/171")["X171_DE_time"].tolist()
# inner_36_3 = io.loadmat("./CWRU/inner/inner36/172")["X172_DE_time"].tolist()
inner_36 = [inner_36_0]
# inner_36 = [inner_36_0, inner_36_1, inner_36_2, inner_36_3]

# inner_54
inner_54_0 = io.loadmat("D:\实验\cwru\CaseWesternReserveUniversityData-master\\12k Drive End Bearing Fault Data\\209")["X209_DE_time"].tolist()
# inner_54_1 = io.loadmat("./CWRU/inner/inner54/210")["X210_DE_time"].tolist()
# inner_54_2 = io.loadmat("./CWRU/inner/inner54/211")["X211_DE_time"].tolist()
# inner_54_3 = io.loadmat("./CWRU/inner/inner54/212")["X212_DE_time"].tolist()
inner_54 = [inner_54_0]
# inner_54 = [inner_54_0, inner_54_1, inner_54_2, inner_54_3]

# outer_18
outer_18_0 = io.loadmat("D:\实验\cwru\CaseWesternReserveUniversityData-master\\12k Drive End Bearing Fault Data\\130")["X130_DE_time"].tolist()
# outer_18_1 = io.loadmat("./CWRU/outer/outer18/131")["X131_DE_time"].tolist()
# outer_18_2 = io.loadmat("./CWRU/outer/outer18/132")["X132_DE_time"].tolist()
# outer_18_3 = io.loadmat("./CWRU/outer/outer18/133")["X133_DE_time"].tolist()
outer_18 = [outer_18_0]
# outer_18 = [outer_18_0, outer_18_1, outer_18_2, outer_18_3]

# outer_36
outer_36_0 = io.loadmat("D:\实验\cwru\CaseWesternReserveUniversityData-master\\12k Drive End Bearing Fault Data\\197")["X197_DE_time"].tolist()
# outer_36_1 = io.loadmat("./CWRU/outer/outer36/198")["X198_DE_time"].tolist()
# outer_36_2 = io.loadmat("./CWRU/outer/outer36/199")["X199_DE_time"].tolist()
# outer_36_3 = io.loadmat("./CWRU/outer/outer36/200")["X200_DE_time"].tolist()
outer_36 = [outer_36_0]
# outer_36 = [outer_36_0, outer_36_1, outer_36_2, outer_36_3]

# outer_54
outer_54_0 = io.loadmat("D:\实验\cwru\CaseWesternReserveUniversityData-master\\12k Drive End Bearing Fault Data\\234")["X234_DE_time"].tolist()
# outer_54_1 = io.loadmat("./CWRU/outer/outer54/235")["X235_DE_time"].tolist()
# outer_54_2 = io.loadmat("./CWRU/outer/outer54/236")["X236_DE_time"].tolist()
# outer_54_3 = io.loadmat("./CWRU/outer/outer54/237")["X237_DE_time"].tolist()
outer_54 = [outer_54_0]
# outer_54 = [outer_54_0, outer_54_1, outer_54_2, outer_54_3]

# normal
normal_0 = io.loadmat("D:\实验\cwru\CaseWesternReserveUniversityData-master\\Normal Baseline Data\\97")["X097_DE_time"].tolist()
normal_1 = io.loadmat("D:\实验\cwru\CaseWesternReserveUniversityData-master\\Normal Baseline Data\\98")["X098_DE_time"].tolist()
normal_2 = io.loadmat("D:\实验\cwru\CaseWesternReserveUniversityData-master\\Normal Baseline Data\\99")["X099_DE_time"].tolist()
normal_3 = io.loadmat("D:\实验\cwru\CaseWesternReserveUniversityData-master\\Normal Baseline Data\\100")["X100_DE_time"].tolist()
normal = [normal_0, normal_1, normal_2, normal_3]

# datasetname = ["12k Drive End Bearing Fault Data", "12k Fan End Bearing Fault Data", "48k Drive End Bearing Fault Data",
#                "Normal Baseline Data"]
# normalname = ["97.mat", "98.mat", "99.mat", "100.mat"]
#
# # label
# label = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # The failure data is labeled 1-9
# axis = ["_DE_time", "_FE_time", "_BA_time"]

# all_data
all_data = [
    normal,
    ball_18,
    ball_36,
    ball_54,
    inner_18,
    inner_36,
    inner_54,
    outer_18,
    outer_36,
    outer_54,
]

normal_imgs = []
inner_18_imgs = []
inner_36_imgs = []
inner_54_imgs = []
outer_18_imgs = []
outer_36_imgs = []
outer_54_imgs = []
ball_18_imgs = []
ball_36_imgs = []
ball_54_imgs = []

for index in range(10):
    data = all_data[index]
    for load_type in range(1):
        load_data = data[load_type]
        max_start = len(load_data) - 4096
        starts = []
        for i in range(500):
            # 随机一个start，不在starts里，就加入
            while True:
                start = random.randint(0, max_start)
                if start not in starts:
                    starts.append(start)
                    break
            # 将4096个数据点转化成64×64的二维图
            temp = load_data[start: start + 4096]
            temp = np.array(temp)
            temp = temp.reshape(64, 64)
            max = -2
            min = 2
            for i in range(64):
                for j in range(64):
                    if (temp[i][j] > max):
                        max = temp[i][j]

                    if (temp[i][j] < min):
                        min = temp[i][j]
            for i in range(64):
                for j in range(64):
                    temp[i][j] = 255 * (temp[i][j] - min) / (max - min)

            if (index == 0):
                normal_imgs.append(temp)
            if (index == 1 ):
                ball_18_imgs.append(temp)
            if ( index == 2 ):
                ball_36_imgs.append(temp)
            if ( index == 3):
                ball_54_imgs.append(temp)

            if (index == 4 ):
                inner_18_imgs.append(temp)
            if ( index == 5 ):
                inner_36_imgs.append(temp)
            if ( index == 6):
                inner_54_imgs.append(temp)

            if (index == 7 ):
                outer_18_imgs.append(temp)
            if ( index == 8 ):
                outer_36_imgs.append(temp)
            if ( index == 9):
                outer_54_imgs.append(temp)


np.savez("normal_imgs", *normal_imgs)
np.savez("ball_18_imgs", *ball_18_imgs)
np.savez("ball_36_imgs", *ball_36_imgs)
np.savez("ball_54_imgs", *ball_54_imgs)
np.savez("inner_18_imgs", *inner_18_imgs)
np.savez("inner_36_imgs", *inner_36_imgs)
np.savez("inner_54_imgs", *inner_54_imgs)
np.savez("outer_18_imgs", *outer_18_imgs)
np.savez("outer_36_imgs", *outer_36_imgs)
np.savez("outer_54_imgs", *outer_54_imgs)