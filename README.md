V叔的炼丹炉
===========

代码还很初步，主要是给大家提供灵感，不保证能跑。

# 思路

- 1. 用selenium从搜索引擎下载图片
- 2. 脚本控制训练

# 环境配置

```
# 下载本repo
git clone --depth 1 https://github.com/vshu3000/danlab
cd danlab

# 进到目录后下载两个主要的包
git clone --depth 1  https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
git clone --depth 1  https://github.com/bmaltais/kohya_ss

# 设置python venv, 就是装那两个包的依赖
python3 -m venv venv
. venv/bin/activate
# 所有的python活动建议都在venv里做
pip3 install -r stable-diffusion-webui/requirements.txt
pip3 install -r kohya_ss/requirements.txt

# 安装别的包，缺啥装啥
pip3 install selenium

# 需要wget
apt-get install wget

```

# 炼丹流程

每个丹在danlab下面创建一个子目录，比如`001_xxx`。

注意一步一步跑，观察输出。

```
mkdir 001_xxx
cd 001_xxx

echo "关键字" > query
../query.py			# 用selenium从duckduckgo下载图片URL
					# 查询关键字就是query文件的内容
../download.sh		# 下载图片到raw目录
../filter.py		# 过滤图片, 好的进filter，坏的进bad
			# 完了最好再手工把filter里不好的图片删掉。
			# 有的黑白图对模型影响很大。
../prep.py		# 用webui的预处理流程，训练数据进samples
../train.sh		# 训练Lora
# 这里要准备一个prompts文件，两行文本
# 第一行是正提示，第二行是负提示，放在本目录下
../make_images.py	# 产生图片，输出到output，随时按ctrl+C结束
```



