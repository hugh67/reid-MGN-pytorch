import data
import loss
import torch
import model
from trainer import Trainer

from option import args
import utils.utility as utility
import os

# os.environ['CUDA_VISIBLE_DEVICES']='0'  # 指定GPU
# print(args)  # after run option.py can get these 参数

ckpt = utility.checkpoint(args)  # args,log,dir,log_file...functions...


# print(ckpt)
# 以下3步可单独测试
loader = data.Data(args)
model = model.Model(args, ckpt)
loss = loss.Loss(args, ckpt) if not args.test_only else None  # if没有冒号

# print(1) if not True else None  # if没有冒号  output:
# print(2) if not False else None  # if没有冒号 output: 2

trainer = Trainer(args, model, loss, loader, ckpt)

n = 0
while not trainer.terminate():
    n += 1
    trainer.train()
    # print("n=")
    # print(n)
    if args.test_every != 0 and n % args.test_every == 0:
        trainer.test()
        # print("hello")
