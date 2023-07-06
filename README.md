# MMHQA-ICL



A simple approach to use LLM to solve question answering over Text, Tables and Images using in-context learning.



## 模块

一共包含四个，问题分类器，图像检索器，文本检索器，问答系统。

问答系统模块利用问题分类器的结果以及图像、文本检索器从给定的文本图像中检索到的内容，利用大模型通过 in-context learning 的方式对问题给出回答。



其中还有图像描述生成器（用 LLaVA 生成 image-caption ），暂时未放上来。只需将对应 image-caption 的 json 文件放到 utils 内即可（已放）。



## 运行

注意修改每个模块 dataset.py 中数据集位置，即 `_DATA_PATH` 

`deberta-large` 权重在 [huggingface](https://huggingface.co/microsoft/deberta-v3-large/tree/main) 上下载，需放在 `ptm/deberta-large` 目录下



### classifier

```bash
PYTHONPATH=$PYTHONPATH:$(pwd) python ./classifier_module/train.py \
--batch-size 4 \
--lr 8e-7 \
--test \
--epoch 5
```



### retriever - image

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=$PYTHONPATH:$(pwd) python ./retriever_module/train.py \
--n-gpu 2 \
--log-file train_image.log \
--image_or_text image \
--lr 5e-6 \
--test \
--epoch 20
```



### retriever - text

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=$PYTHONPATH:$(pwd) python ./retriever_module/train.py \
--n-gpu 2 \
--test \
--lr 6e-6 \
--image_or_text text \
--epoch 5
```



### question-answering

oracle配置（使用 oracle-classifier + oracle-retriever）：

```bash
PYTHONPATH=$PYTHONPATH:$(pwd) python ./run.py --dataset mmqa \
--dataset_split validation \
--prompt_file templates/prompt.json \
--n_parallel_prompts 1 \
--n_processes 1 \
--temperature 0.4 \
--engine "text-davinci-003" \
--max_api_total_tokens 4200 \
--oracle-classifier \
--oracle-retriever \
--retriever dpmlb
```



非 oracle 配置（使用训练的 classifier 和 retriever）

```bash
PYTHONPATH=$PYTHONPATH:$(pwd) python ./run.py --dataset mmqa \
--dataset_split validation \
--prompt_file templates/prompt.json \
--n_parallel_prompts 1 \
--n_processes 1 \
--temperature 0.4 \
--engine "text-davinci-003" \
--max_api_total_tokens 4200 \
--retriever dpmlb
```

