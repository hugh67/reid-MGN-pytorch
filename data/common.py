import os
import re

# 返回图像列表
def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    # 断言是否存在这样的目录
    assert os.path.isdir(directory), 'dataset is not exists!{}'.format(directory)
    return sorted([os.path.join(root, f)
                   for root, _, files in os.walk(directory) for f in files
                   if re.match(r'([\w]+\.(?:' + ext + '))', f)])
