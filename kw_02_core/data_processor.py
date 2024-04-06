import numpy as np
from pysteps import motion
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import netCDF4 as nc
import os



class Data(Dataset):
    # 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)

    def get_test_data(self, seq_len, normalise):

        # 建立滑块（序列seq）数据
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])    # 滑窗的时候是从下标1开始的，略去了第0个样本（1-seq_len）

        data_windows = np.array(data_windows).astype(float)
        # 只对x（特征值）进行归一化
        # x = self.normalise_windows(data_windows[:, :-1, :-1], single_window=False) if normalise else data_windows
        # 同时对x,y（特征值和目标值）均进行归一化
        # data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows # （15,8,13）
        x = data_windows[:, :, :-1]
        y = data_windows[:, -1, -1]
        return x, y

    def get_train_data(self, seq_len, normalise):

        # 建立滑块（序列seq）数据
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        window = self.data_train[i:i+seq_len]
        # window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        # 将一个序列中所有的特征值作为x，最后一个目标值作为y
        x = window[:, :-1]
        y = window[-1, -1].reshape(1,1)
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                try:
                    normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]     # 此时的归一化方法和DataLoader'__init__'中的归一化中选取一个，此处是基于相对滑动序列的初始值的变幅归一的
                except:
                    print('第0行，第%s列：' % col_i, float(window[0, col_i]))
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T 
            normalised_data.append(normalised_window)
        return np.array(normalised_data)


class Data_GPM(Dataset):
    # 初始化函数，得到数据
    def __init__(self, configs):
        self.configs = configs

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)

    def seq_windows(self, data, i, seq_len):
        """
        滑动窗口建立序列数据→多个输入
        :param data: 输入的待处理数据
        :param i: 第i个滑块
        :param seq_len: 滑块序列长度
        :return: 返回序列数据（特征+目标）xi，yi
        """
        window = data[i:i + seq_len]
        # window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        # 将一个序列中前9个作为x，第10个作为y，后8个先不管
        xi = window[:9, :, :]
        yi = window[9:12, :, :]
        return xi, yi

    def seq_windows_iter(self, data, i, seq_len):
        """
        滑动窗口建立序列数据→多个输入
        :param data: 输入的待处理数据
        :param i: 第i个滑块
        :param seq_len: 滑块序列长度
        :return: 返回序列数据（特征+目标）xi，yi
        """
        window = data[i:i + seq_len]
        # window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        # 将一个序列中前9个作为x，第10个作为y，后8个先不管
        xi = window[:9, :, :]
        yi = window[9:, :, :]
        return xi, yi

    def load_data(self, file_n):
        """
        加载数据文件路径，依次读取，转换成x，y
        :param num_data: 需要加载的数据量
        :return: 将返回值x, y赋给self.data, self.label
        """
        num_read = 0
        dataset = []
        for file_name in file_n:
            # if file_name.endswith("nc"):
            file = self.configs["data"]["file_path"] + file_name
            data = nc.Dataset(file)  # open the dataset
            precip_data = data.variables['precipitationCal'][0].data
            precip_data[precip_data < 0] = 0

            dataset.append(precip_data)

            # dataset = data.variables['precip']
            num_read += 1

        R = np.array(dataset)

        R[np.isnan(R)] = 0 # 缺失值处理：-9999.9为缺失值

        # 建立滑块，生成序列x,y;划分数据集→train、verif、test
        seq_len = self.configs['data']['sequence_length']
        data_x, data_y = [], []
        for i in range(0, len(R), seq_len):
            xi, yi = self.seq_windows(R, i, seq_len)
            data_x.append(xi)
            data_y.append(yi)
        x, y = np.array(data_x), np.array(data_y)
        self.data = x
        self.label = y


    def load_wdata(self, file_n):
        """
               加载数据文件路径，依次读取，转换成x，y
               :param num_data: 需要加载的数据量
               :return: 将返回值x, y赋给self.data, self.label
               """
        num_read = 0
        dataset = []
        for file_name in file_n:
            # if file_name.endswith("nc"):
            file = self.configs["data"]["window_path"] + file_name
            precip_data = np.load(file)  # open the dataset
            precip_data[precip_data < 0] = 0

            dataset.append(precip_data)

            # dataset = data.variables['precip']
            n_leadtimes = self.configs['data']['n_leadtimes']
            num_read += 1

        R = np.array(dataset)

        R[np.isnan(R)] = 0  # 缺失值处理：-9999.9为缺失值

        # 建立滑块，生成序列x,y;划分数据集→train、verif、test
        seq_len = self.configs['data']['sequence_length']
        data_x, data_y = [], []
        for i in range(len(R)):
            data_x.append(R[i][:9, :, :])
            data_y.append(R[i][9:12, :, :])
        x, y = np.array(data_x), np.array(data_y)
        self.data = x
        self.label = y

    def load_test(self, num_data):
        """
               加载数据文件路径，依次读取，转换成x，y
               :param num_data: 需要加载的数据量
               :return: 将返回值x, y赋给self.data, self.label
               """
        num_read = 0
        dataset = []
        for file_name in os.listdir(self.configs["eval"]["test_path"]):
            # if file_name.endswith("nc"):
            file = self.configs["eval"]["test_path"] + file_name
            data = nc.Dataset(file)  # open the dataset
            precip_data = data.variables['precipitationCal'][0].data
            lat = data.variables['lat'][:]
            precip_data[precip_data < 0] = 0
            precip_data[np.isnan(precip_data)] = 0
            dataset.append(precip_data)
            # dataset = data.variables['precip']
            num_read += 1
            if num_read >= num_data:
                print(data.variables[
                          'precipitationCal'])  # 查看数据集的结构，属性dict_keys(['datetime', 'lon', 'lat', 'crs', 'precip'])
                break
            else:
                continue
        R = np.array(dataset)
        R[np.isnan(R)] = 0 # 缺失值处理：-9999.9为缺失值
        # 筛选降雨量超过一定阈值的降水
        # pre_limit = 10
        # R[R < pre_limit] = 0

        # 建立滑块，生成序列x,y;划分数据集→train、verif、test
        seq_len = self.configs['data']['sequence_length']
        data_x, data_y = [], []
        for i in range(0, len(R), seq_len):
            xi, yi = self.seq_windows(R, i, seq_len)
            data_x.append(xi)
            data_y.append(yi)
        x, y = np.array(data_x), np.array(data_y)
        return x, y

    def load_test_iter(self, num_data):
        """
               加载数据文件路径，依次读取，转换成x，y
               :param num_data: 需要加载的数据量
               :return: 将返回值x, y赋给self.data, self.label
               """
        num_read = 0
        dataset = []
        for file_name in os.listdir(self.configs["eval"]["test_path"]):
            # if file_name.endswith("nc"):
            file = self.configs["eval"]["test_path"] + file_name
            data = nc.Dataset(file)  # open the dataset
            precip_data = data.variables['precipitationCal'][0].data
            lat = data.variables['lat'][:]
            precip_data[precip_data < 0] = 0
            precip_data[np.isnan(precip_data)] = 0
            dataset.append(precip_data)
            # dataset = data.variables['precip']
            num_read += 1
            if num_read >= num_data:
                print(data.variables[
                          'precipitationCal'])  # 查看数据集的结构，属性dict_keys(['datetime', 'lon', 'lat', 'crs', 'precip'])
                break
            else:
                continue
        R = np.array(dataset)
        R[np.isnan(R)] = 0 # 缺失值处理：-9999.9为缺失值
        # 筛选降雨量超过一定阈值的降水
        # pre_limit = 10
        # R[R < pre_limit] = 0

        # 建立滑块，生成序列x,y;划分数据集→train、verif、test
        seq_len = self.configs['data']['sequence_length']
        data_x, data_y = [], []
        for i in range(0, len(R), seq_len):
            xi, yi = self.seq_windows_iter(R, i, seq_len)
            data_x.append(xi)
            data_y.append(yi)
        x, y = np.array(data_x), np.array(data_y)
        return x, y

    def load_event(self, num_data, event_path):
        """
        加载数据文件路径，依次读取，转换成x，y
        :param num_data: 需要加载的数据量
        :param event_path: 需要加载的降水事件文件路径
        :return: 将返回值x, y赋给self.data, self.label
        """
        num_read = 0
        dataset = []
        for file_name in os.listdir(event_path):
            # if file_name.endswith("nc"):
            file = event_path + file_name
            data = nc.Dataset(file)  # open the dataset
            precip_data = data.variables['precipitationCal'][0].data
            lat = data.variables['lat'][:]
            precip_data[precip_data < 0] = 0
            precip_data[np.isnan(precip_data)] = 0
            dataset.append(precip_data)
            # dataset = data.variables['precip']
            num_read += 1
            if num_read >= num_data:
                print(data.variables[
                          'precipitationCal'])  # 查看数据集的结构，属性dict_keys(['datetime', 'lon', 'lat', 'crs', 'precip'])
                break
            else:
                continue
        R = np.array(dataset)
        R[np.isnan(R)] = 0 # 缺失值处理：-9999.9为缺失值
        # 筛选降雨量超过一定阈值的降水
        # pre_limit = 10
        # R[R < pre_limit] = 0

        # 建立滑块，生成序列x,y;划分数据集→train、verif、test
        seq_len = self.configs['data']['sequence_length']
        data_x, data_y = [], []
        for i in range(0, len(R), seq_len):
            xi, yi = self.seq_windows_iter(R, i, seq_len)
            data_x.append(xi)
            data_y.append(yi)
        x, y = np.array(data_x), np.array(data_y)
        return x, y

    def load_verif(self, num_data):
        """
               加载数据文件路径，依次读取，转换成x，y
               :param num_data: 需要加载的数据量
               :return: 将返回值x, y赋给self.data, self.label
               """
        num_read = 0
        dataset = []
        for file_name in os.listdir(self.configs["verif"]["verif_path"]):
            # if file_name.endswith("nc"):
            file = self.configs["verif"]["verif_path"] + file_name
            data = nc.Dataset(file)  # open the dataset
            precip_data = data.variables['precipitationCal'][0].data
            precip_data[precip_data < 0] = 0
            precip_data[np.isnan(precip_data)] = 0
            dataset.append(precip_data)
            # dataset = data.variables['precip']
            num_read += 1
            if num_read >= num_data:
                print(data.variables[
                          'precipitationCal'])  # 查看数据集的结构，属性dict_keys(['datetime', 'lon', 'lat', 'crs', 'precip'])
                break
            else:
                continue
        R = np.array(dataset)
        R[np.isnan(R)] = 0 # 缺失值处理：-9999.9为缺失值
        # 筛选降雨量超过一定阈值的降水
        # pre_limit = 10
        # R[R < pre_limit] = 0

        # 建立滑块，生成序列x,y;划分数据集→train、verif、test
        seq_len = self.configs['data']['sequence_length']
        data_x, data_y = [], []
        for i in range(0, len(R), seq_len):
            xi, yi = self.seq_windows(R, i, seq_len)
            data_x.append(xi)
            data_y.append(yi)
        x, y = np.array(data_x), np.array(data_y)
        return x, y

    def data_split(self):
        """
        :return:训练集、验证集、测试集的特征值和目标值：x_train, x_verif, x_test, y_train, y_verif, y_test
        """
        # 增加一个通道数的维度，用于卷积输入
        x, y = self.data[:, np.newaxis, :, :, :], self.label[:, np.newaxis, np.newaxis, :, :]
        x_train_verif, x_test, y_train_verif, y_test = train_test_split(x, y, random_state=1, test_size=1-self.configs["data"]["train_test_split"])
        trian_len = round(len(x_train_verif)*self.configs["data"]["train_verif_split"])
        x_train = x_train_verif[:trian_len]
        y_train = y_train_verif[:trian_len]
        x_verif = x_train_verif[trian_len:]
        y_verif = y_train_verif[trian_len:]
        return x_train, x_verif, x_test, y_train, y_verif, y_test

    def data_add_OF(self):
        """
        :return:没有返回值，将self.data更新→添加光流法计算的motion_vector
        """
        x_new = []
        # 4、Lucas-Kanade (LK)
        oflow_method = motion.get_method("LK")
        for s in range(len(self.data)):
            R = self.data[s]
            V1 = oflow_method(R, verbose=True)
            x_newi = np.append(R, V1, axis=0)
            x_new.append((x_newi))
        if x_new != None:
            self.data = np.array(x_new)


    def clip_quarter(self, data_ori):
        shape = data_ori.shape
        data_clip = data_ori[:, :, shape[2]//2:, :shape[3]//2]
        return data_clip




