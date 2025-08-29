此项目用于在MMLU数据集上评测大模型的论文复现
需下载MMLU数据集
其中包含将csv格式的数据集合并整理成json文件的mmlu_merge.py
和用于评测大模型的文件qy_eval.py
下载数据集指令：
# 下载官方 MMLU 数据集
wget -c https://people.eecs.berkeley.edu/~hendrycks/data.tar
# 解压
tar -xf data.tar

下载完成后依次执行代码即可：
python merge_mmlu.py
python qy_eval.py
