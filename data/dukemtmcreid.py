from data.common import list_pictures  # data/common.py 中的list_pictures function
from torch.utils.data import dataset
from torchvision.datasets.folder import default_loader


class DukeMTMCreID(dataset.Dataset):
    def __init__(self, args, transform, dtype):

        self.transform = transform
        self.loader = default_loader

        data_path = args.datadir
        if dtype == 'train':
            data_path += '/bounding_box_train'
        elif dtype == 'test':
            data_path += '/bounding_box_test'
        else:
            data_path += '/query'

        # 图像数据
        self.imgs = [path for path in list_pictures(data_path) if self.id(path) != -1]
        # 遍历
        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}

    def __getitem__(self, index):  # 解决如何读数据
        path = self.imgs[index]
        target = self._id2label[self.id(path)]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):  # 放回数据集的长度
        return len(self.imgs)


# 以 0001_c2_f0046182.jpg 为例
# 1） 0001 表示每个人的标签编号；
# 2） c2 表示来自第二个摄像头(camera2)，共有 8 个摄像头；
# 3） f0046182 表示来自第二个摄像头的第 46182 帧。

    @staticmethod
    def id(file_path):  # 返回每个人的标签编号，从0001到1501
        """
        :param file_path: unix style file path
        :return: person id
        """
        # return int(file_path.split('\\')[-1].split('_')[0]) # 适合于Windows
        return int(file_path.split('/')[-1].split('_')[0])


    @staticmethod
    def camera(file_path):  # 返回摄像头的id
        """
        :param file_path: unix style file path
        :return: camera id
        """
        return int(file_path.split('/')[-1].split('_')[1][1])

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return sorted(set(self.ids))

    @property
    def cameras(self):
        """
        :return: camera id list corresponding to dataset image paths
        """
        return [self.camera(path) for path in self.imgs]
