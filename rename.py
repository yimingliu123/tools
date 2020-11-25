import os
import glob

data_dir = "/home/lym/srdata_path/GOPRO"
dir_hr = "/home/lym/srdata_path/GOPRO/GOPRO_blur"
dir_lr = "/home/lym/srdata_path/GOPRO/GOPRO_sharp"
names_hr = sorted(
    glob.glob(os.path.join(dir_hr, '*' + ".png"))  ####dir_hr改成dir_lr
)
print(names_hr)
for f in names_hr:
    filename, _ = os.path.splitext(os.path.basename(f))
    filename = filename.split('_')
    filename = filename[0] + "_" + filename[1] + "_" + filename[2]
    print("filename_split")
    print(filename)
    s = 4
    LR = os.path.join(
        dir_lr, 'X{}/{}_{}_LR{}'.format(
            s, filename, s, ".png"))
    LR_new = os.path.join(
        dir_lr, 'X{}/{}_{}{}'.format(
            s, filename, s, ".png"))
    os.rename(LR,LR_new)