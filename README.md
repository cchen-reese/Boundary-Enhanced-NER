# Boundary-Enhanced-NER
It's keeping updating.

# env
pytorch 0.4.1
gpu

# dataset
1. OntoNotes V4/V5 word+char
2. Weibo word+char https://github.com/cchen-reese/weiboNER

# embedding
We used Mixed-large 综合(Baidu Netdisk / Google Drive)，300dim
https://github.com/Embedding/Chinese-Word-Vectors

# dependency parser
ddparser   https://pypi.org/project/ddparser/

# train
python main.py --config train.config
