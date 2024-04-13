# MMHQA-ICL

A simple approach to use LLM to solve question answering over Text, Tables and Images using in-context learning.

## Modules

There are a total of four modules: a question classifier, an image retriever, a text retriever, and a question-answering module.

The question-answering module utilizes the results from the question classifier and the content retrieved from the given text and images by the image and text retrievers. It then employs a large language model to provide answers to the questions through in-context learning.

## Run

Make sure to modify the dataset location in each module's `dataset.py` file.

Download the model weights of `deberta-large` from [huggingface](https://huggingface.co/microsoft/deberta-v3-large/tree/main) and put them in `ptm/deberta-large`.

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

oracle settings（oracle-classifier + oracle-retriever）：

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


non-oracle settings（using trained classifier and retriever）

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

