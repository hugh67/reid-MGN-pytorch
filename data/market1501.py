from data.common import list_pictures  # data/common.py 中的list_pictures function
from torch.utils.data import dataset
from torchvision.datasets.folder import default_loader


class Market1501(dataset.Dataset):
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


# 命名规则
# 以 0001_c1s1_000151_01.jpg 为例
# 1） 0001 表示每个人的标签编号，从0001到1501；
# 2） c1 表示第一个摄像头(camera1)，共有6个摄像头；
# 3） s1 表示第一个录像片段(sequece1)，每个摄像机都有数个录像段；
# 4） 000151 表示 c1s1 的第000151帧图片，视频帧率25fps；
# 5） 01 表示 c1s1_001051 这一帧上的第1个检测框，由于采用DPM检测器，对于每一帧上的行人可能会框出好几个bbox。00 表示手工标注框

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
