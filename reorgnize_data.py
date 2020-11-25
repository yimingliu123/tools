import os
import glob

class RenameData():
    def __init__(self):
        self.train_hr_path = "/home/lym/srdata_path/GOPRO/GOPRO_blur"
        self.train_lr_X1_path = "/home/lym/srdata_path/GOPRO/GOPRO_sharp/X1" ###x1 放1280*720
        self.train_lr_X2_path = "/home/lym/srdata_path/GOPRO/GOPRO_sharp/X2"#####X2 缩小1/2
        self.train_lr_X4_path = "/home/lym/srdata_path/GOPRO/GOPRO_sharp/X4"#####X2 缩小1/4

        self.test_hr_path = "/home/lym/srdata_path/benchmark/GOPRO/GOPRO_blur"
        self.test_lr_X1_path = "/home/lym/srdata_path/benchmark/GOPRO/sharp_bicubic/X1"###x1 放1280*720
        self.test_lr_X2_path = "/home/lym/srdata_path/benchmark/GOPRO/sharp_bicubic/X2"#####X2 缩小1/2
        self.test_lr_X4_path = "/home/lym/srdata_path/benchmark/GOPRO/sharp_bicubic/X4"#####X2 缩小1/4

    def rename_train_hr(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.train_hr_path,"*", '*.png' ))
        )
        print("names_hr")
        print(names_hr)
        # names_lr = [[] for _ in scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            print(filename)
            train_hr_path_list = f.split('/')
            each_dir = train_hr_path_list[-2]
            print(each_dir)

            dst = os.path.join(self.train_hr_path,  each_dir+'_'+filename+'.png' )
            ##os.rename(f,dst)

    def rename_train_lr_X1(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.train_lr_X1_path, "*", '*.png'))
        )
        print("names_hr")
        print(names_hr)
        # names_lr = [[] for _ in scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            #print(filename)
            train_hr_path_list = f.split('/')
            each_dir = train_hr_path_list[-2]
            #print(each_dir)

            dst = os.path.join(self.train_lr_X1_path, each_dir + '_' + filename + '.png')
            #print(dst)
            ##os.rename(f, dst)

    def rename_train_lr_X2(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.train_lr_X2_path, "*", '*.png'))
        )
        print("names_hr")
        print(names_hr)
        # names_lr = [[] for _ in scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            #print(filename)
            train_hr_path_list = f.split('/')
            each_dir = train_hr_path_list[-2]
            #print(each_dir)

            dst = os.path.join(self.train_lr_X2_path, each_dir + '_' + filename + '.png')
            #print(dst)
            ##os.rename(f, dst)

    def rename_train_lr_X4(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.train_lr_X4_path, "*", '*.png'))
        )
        print("names_hr")
        print(names_hr)
        # names_lr = [[] for _ in scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            # print(filename)
            train_hr_path_list = f.split('/')
            each_dir = train_hr_path_list[-2]
            # print(each_dir)

            dst = os.path.join(self.train_lr_X4_path, each_dir + '_' + filename + '.png')
            # print(dst)
            os.rename(f, dst)

    def rename_test_hr(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.test_hr_path,"*", '*.png' ))
        )
        print("names_hr")
        print(names_hr)
        # names_lr = [[] for _ in scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            print(filename)
            train_hr_path_list = f.split('/')
            each_dir = train_hr_path_list[-2]
            print(each_dir)

            dst = os.path.join(self.test_hr_path,  each_dir+'_'+filename+'.png' )
            os.rename(f,dst)

    def rename_test_lr_X1(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.test_lr_X1_path, "*", '*.png'))
        )
        print("names_hr")
        print(names_hr)
        # names_lr = [[] for _ in scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            #print(filename)
            train_hr_path_list = f.split('/')
            each_dir = train_hr_path_list[-2]
            #print(each_dir)

            dst = os.path.join(self.test_lr_X1_path, each_dir + '_' + filename + '.png')
            #print(dst)
            os.rename(f, dst)

    def rename_test_lr_X2(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.test_lr_X2_path, "*", '*.png'))
        )
        print("names_hr")
        print(names_hr)
        # names_lr = [[] for _ in scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            #print(filename)
            train_hr_path_list = f.split('/')
            each_dir = train_hr_path_list[-2]
            #print(each_dir)

            dst = os.path.join(self.test_lr_X2_path, each_dir + '_' + filename + '.png')
            #print(dst)
            os.rename(f, dst)

    def rename_test_lr_X4(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.test_lr_X4_path, "*", '*.png'))
        )
        print("names_hr")
        print(names_hr)
        # names_lr = [[] for _ in scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            # print(filename)
            train_hr_path_list = f.split('/')
            each_dir = train_hr_path_list[-2]
            # print(each_dir)

            dst = os.path.join(self.test_lr_X4_path, each_dir + '_' + filename + '.png')
            # print(dst)
            os.rename(f, dst)



####one by one 小心使用
# RenameData().rename_train_lr_X4()
# RenameData().rename_test_hr()
# RenameData().rename_test_lr_X1()
# RenameData().rename_test_lr_X2()
RenameData().rename_test_lr_X4()