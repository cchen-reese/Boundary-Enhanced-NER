# Boundary-Enhanced-NER
This is for ACL2021 paper, https://aclanthology.org/2021.acl-short.4/.
It's keeping updating.

# env
pytorch 0.4.1
gpu

# dataset
1. OntoNotes V4/V5 word+char
2. Weibo word+char the dataset is in data named train/dev/test_word_bmes_pos_ddenp.txt. You can try to run our project with this dataset.

# embedding
We used Mixed-large 综合(Baidu Netdisk / Google Drive)，300dim.
https://github.com/Embedding/Chinese-Word-Vectors
Warning: you must put the embedding file in data, with the right setting in train.config.

# dependency parser
ddparser   https://pypi.org/project/ddparser/

# train
python main.py --config train.config
