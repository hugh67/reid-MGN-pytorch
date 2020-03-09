from importlib import import_module  # 动态导入对象
from torchvision import transforms  # 数据处理工具
from utils.random_erasing import RandomErasing  # 随机擦除
from data.sampler import RandomSampler  # 数据取样
from torch.utils.data import dataloader  # 数据读取


class Data:
    def __init__(self, args):
        print('[INFO] Making Data...')
        train_list = [
            transforms.Resize((args.height, args.width), interpolation=3),
            # resize the picture （interpolation=3为选择插值方法）
            transforms.RandomHorizontalFlip(),  # 依概率p水平翻转
            transforms.ToTensor(),  # 转Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
        ]
        if args.random_erasing:  # 随机擦除
            train_list.append(RandomErasing(probability=args.probability, mean=[0.0, 0.0, 0.0]))

        train_transform = transforms.Compose(train_list)  # 组合步骤

        test_transform = transforms.Compose([
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if not args.test_only:
            # 加载data/market1501.py，cuhk03.py, dukemtmcreid.py
            module_train = import_module('data.' + args.data_train.lower())  # lower() aid : A to a
            # getattr(对象，属性)为获取某一对象的某个属性的属性值
            # def __init__(self, args = args, transform = train_transform, dtype = 'train'):
            self.trainset = getattr(module_train, args.data_train)(args, train_transform, 'train')
            self.train_loader = dataloader.DataLoader(self.trainset,  # 传入的数据集
                                                      # 自定义从数据集中取样本的策略，如果指定这个参数，那么shuffle必须为False
                                                      sampler=RandomSampler(self.trainset, args.batchid,
                                                                            batch_image=args.batchimage),
                                                      shuffle=False,  # 在每个epoch开始的时候，是否对数据进行重新排序
                                                      batch_size=args.batchid * args.batchimage,  # 每个batch有多少个样本
                                                      # 这个参数决定了有几个进程来处理data loading。0意味着所有的数据都会被load进主进程
                                                      num_workers=args.nThread)
        else:
            self.train_loader = None

        if args.data_test in ['Market1501']:
            module = import_module('data.' + args.data_train.lower())
            self.testset = getattr(module, args.data_test)(args, test_transform, 'test')
            self.queryset = getattr(module, args.data_test)(args, test_transform, 'query')

        elif args.data_test in ['DukeMTMCreID']:
            module = import_module('data.' + args.data_train.lower())
            self.testset = getattr(module, args.data_test)(args, test_transform, 'test')
            self.queryset = getattr(module, args.data_test)(args, test_transform, 'query')
        else:
            raise Exception()

        self.test_loader = dataloader.DataLoader(self.testset, batch_size=args.batchtest, num_workers=args.nThread)
        self.query_loader = dataloader.DataLoader(self.queryset, batch_size=args.batchtest, num_workers=args.nThread)
