import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import types
from peft import PeftModel
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import tensorflow as tf
import math
from functools import partial
tf.get_logger().setLevel('ERROR')
import matplotlib
import matplotlib.pyplot as plt
from collections import OrderedDict
matplotlib.use('Agg')
import seaborn as sns
from datasets import load_dataset
from peft import PromptTuningConfig, TaskType, PromptTuningInit, get_peft_model, PeftModel, PromptEncoderConfig, \
    PromptEncoderReparameterizationType, PrefixTuningConfig, LoraConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, TrainingArguments, \
    DataCollatorForSeq2Seq, Trainer, PreTrainedModel, RobertaForSequenceClassification, RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention
from transformers.activations import GELUActivation
from transformers.modeling_outputs import SequenceClassifierOutput
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from tqdm import tqdm
import shutil
import time
from torch.nn.utils.rnn import pad_sequence
import tempfile
import psutil
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Tuple, Union

# 自定义的 RobertaSelfAttention，将 softmax 替换为 nn.Softmax
class CustomRobertaSelfAttention(RobertaSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type)
        # 基础参数无需梯度
        self.query.weight.requires_grad = False
        self.query.bias.requires_grad = False
        self.key.weight.requires_grad = False
        self.key.bias.requires_grad = False
        self.value.weight.requires_grad = False
        self.value.bias.requires_grad = False
        # 注意：不要添加任何新的可训练参数！
        self.tee_head_idx = None  # 只是普通属性，不会影响参数加载
        self.layer_name = ""  # 同上
        self.lora_grad_buffer = {} # 暂存lora梯度
        self.lora_A = None
        self.lora_B = None
        self.lora_query = None
        self.lora_key = None
        self.lora_value = None
        self.cached_query_input = None# 缓存输入用于梯度计算
        self.cached_key_input = None
        self.cached_value_input = None
        self.current_qkv_type = None  # 可以是'query', 'key'或'value'
        # del self.query, self.key, self.value

    # def load_peft_state_dict(self, state_dict):
    #     """专门处理PEFT改造后的状态字典"""
    #     # 手动映射参数
    #     param_mapping = {
    #         'query.base_layer.weight': 'query.weight',
    #         'query.base_layer.bias': 'query.bias',
    #         'key.base_layer.weight': 'key.weight',
    #         'key.base_layer.bias': 'key.bias',
    #         'value.base_layer.weight': 'value.weight',
    #         'value.base_layer.bias': 'value.bias'
    #     }
    #
    #     new_state_dict = {}
    #     for peft_key, orig_key in param_mapping.items():
    #         if peft_key in state_dict:
    #             new_state_dict[orig_key] = state_dict[peft_key]
    #
    #     # 加载基础参数
    #     super().load_state_dict(new_state_dict, strict=False)
    #
    #     # 保存LoRA参数
    #     self.lora_params = {
    #         'query': {
    #             'A': state_dict['query.lora_A.default.weight'],
    #             'B': state_dict['query.lora_B.default.weight']
    #         },
    #         'key':{
    #             'A': state_dict['key.lora_A.default.weight'],
    #             'B': state_dict['key.lora_B.default.weight']
    #         },
    #         'value':{
    #             'A': state_dict['value.lora_A.default.weight'],
    #             'B': state_dict['value.lora_B.default.weight']
    #         }
    #     }
    def register_lora_hooks(self):
        if self.lora_query is not None:
            print(f"注册钩子到{self.layer_name}.query")
            self.lora_query.lora_A.default.weight.requires_grad_(True)
            self.lora_query.lora_B.default.weight.requires_grad_(True)
            # self.lora_query.lora_A.default.weight.register_hook(self.backward_hook)
            # self.lora_query.lora_B.default.weight.register_hook(self.backward_hook)
        if self.lora_key is not None:
            self.lora_query.lora_A.default.weight.requires_grad_(True)
            self.lora_query.lora_B.default.weight.requires_grad_(True)
            # self.lora_key.lora_A.default.weight.register_hook(self.backward_hook)
            # self.lora_key.lora_B.default.weight.register_hook(self.backward_hook)
        if self.lora_value is not None:
            self.lora_query.lora_A.default.weight.requires_grad_(True)
            self.lora_query.lora_B.default.weight.requires_grad_(True)
            # self.lora_value.lora_A.default.weight.register_hook(self.backward_hook)
            # self.lora_value.lora_B.default.weight.register_hook(self.backward_hook)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # 确保输入是张量（兼容某些老版本transformers）
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        hidden_states = hidden_states.to(self.query.weight.device)
        # 确保 cached_input 是张量
        if isinstance(hidden_states, (dict, OrderedDict)):
            hidden_states = hidden_states['input_tensor']  # 或其他适当的键

        # 原始投影计算
        ori_query = self.query(hidden_states)
        ori_key = self.key(hidden_states)
        ori_value = self.value(hidden_states)
        # 将lora移入TEE
        tee_query_input = {
            "hidden_states": hidden_states,
            "layer_name": f"base_model.model.roberta.encoder.{self.layer_name}.attention.self.query",
            "layer_type": "lora"
        }
        torch.save(tee_query_input, os.path.join(gradient_dir, f"tee_input_base_model.model.roberta.encoder.{self.layer_name}.attention.self.query.lora.pt"))
        # 等待tee返回计算结果
        while True:
            tee_output_files = os.listdir(output_dir)
            if f"tee_output_base_model.model.roberta.encoder.{self.layer_name}.attention.self.query.lora.pt" in tee_output_files:
                # print(f"tee output for layer {layer_name} received from TEE")
                break

        # 等待TEE返回结果
        tee_output_path = os.path.join(output_dir,
                                       f"tee_output_base_model.model.roberta.encoder.{self.layer_name}.attention.self.query.lora.pt")
        tee_query_output = load_with_retry(tee_output_path)
        os.remove(tee_output_path)
        self.cached_query_input = hidden_states.detach().clone()

        if self.lora_query is not None:
            # 分步计算：dropout -> A -> B
            dropped = self.lora_query.lora_dropout['default'](hidden_states)
            lora_A = self.lora_query.lora_A['default'](dropped)  # shape: [batch, seq, r]
            lora_B = self.lora_query.lora_B['default'](lora_A)  # shape: [batch, seq, out_dim]
            scaling = self.lora_query.scaling["default"]  # 通常为 lora_alpha / r
            lora_output = (lora_B * scaling)

            lora_query = bridge_lora(
                     tee_query_output["encrypted_output"],
                     lora_output
                     )

            query = ori_query + lora_query

            if self.training:
                query.register_hook(lambda grad: self.backward_hook(grad, 'query'))  # 梯度的钩子

        tee_key_input = {
            "hidden_states": hidden_states.clone().cpu(),
            "layer_name": f"base_model.model.roberta.encoder.{self.layer_name}.attention.self.key",
            "layer_type": "lora"
        }
        torch.save(tee_key_input, os.path.join(gradient_dir, f"tee_input_base_model.model.roberta.encoder.{self.layer_name}.attention.self.key.lora.pt"))
        # 等待tee返回计算结果
        while True:
            tee_output_files = os.listdir(output_dir)
            if f"tee_output_base_model.model.roberta.encoder.{self.layer_name}.attention.self.key.lora.pt" in tee_output_files:
                # print(f"tee output for layer {layer_name} received from TEE")
                break

        # 等待TEE返回结果
        tee_output_path = os.path.join(output_dir,
                                       f"tee_output_base_model.model.roberta.encoder.{self.layer_name}.attention.self.key.lora.pt")
        tee_key_output = load_with_retry(tee_output_path)
        os.remove(tee_output_path)

        self.cached_key_input = hidden_states.detach().clone()

        if self.lora_key is not None:

            dropped = self.lora_key.lora_dropout['default'](hidden_states)
            lora_A = self.lora_key.lora_A['default'](dropped)  # shape: [batch, seq, r]
            lora_B = self.lora_key.lora_B['default'](lora_A)  # shape: [batch, seq, out_dim]
            scaling = self.lora_key.scaling["default"]  # 通常为 lora_alpha / r
            lora_output = (lora_B * scaling)

            lora_key = bridge_lora(
                tee_key_output["encrypted_output"],
                lora_output
            )
            key = ori_key + lora_key
            if self.training:
                key.register_hook(lambda grad: self.backward_hook(grad, 'key'))  # 梯度的钩子

        tee_value_input = {
            "hidden_states": hidden_states.clone().cpu(),
            "layer_name": f"base_model.model.roberta.encoder.{self.layer_name}.attention.self.value",
            "layer_type": "lora"
        }
        torch.save(tee_value_input, os.path.join(gradient_dir, f"tee_input_base_model.model.roberta.encoder.{self.layer_name}.attention.self.value.lora.pt"))
        # 等待tee返回计算结果
        while True:
            tee_output_files = os.listdir(output_dir)
            if f"tee_output_base_model.model.roberta.encoder.{self.layer_name}.attention.self.value.lora.pt" in tee_output_files:
                # print(f"tee output for layer {layer_name} received from TEE")
                break

        # 等待TEE返回结果
        tee_output_path = os.path.join(output_dir,
                                       f"tee_output_base_model.model.roberta.encoder.{self.layer_name}.attention.self.value.lora.pt")
        tee_value_output = load_with_retry(tee_output_path)
        os.remove(tee_output_path)
        #
        self.cached_value_input = hidden_states.detach().clone()

        if self.lora_value is not None:
            dropped = self.lora_value.lora_dropout['default'](hidden_states)
            lora_A = self.lora_value.lora_A['default'](dropped)  # shape: [batch, seq, r]
            lora_B = self.lora_value.lora_B['default'](lora_A)  # shape: [batch, seq, out_dim]
            scaling = self.lora_value.scaling["default"]  # 通常为 lora_alpha / r
            lora_output = (lora_B * scaling)
            print((f"self.lora_value.lora_A['default']: {self.lora_value.lora_A['default'].weight.norm().item()}"))
            print(f"lora_A: {lora_A.detach().norm().item()}, lora_B: {lora_B.detach().norm().item()}")
            lora_value = bridge_lora(
                tee_value_output["encrypted_output"],
                lora_output
            )
            print(f"new: {tee_value_output['encrypted_output'].detach().norm().item()}, raw:{lora_output.detach().norm().item()}")
            value = ori_value + lora_value # ***
            if self.training:
                value.register_hook(lambda grad: self.backward_hook(grad, 'value'))  # 梯度的钩子

        # 分割多头 [batch_size, seq_len, num_heads, head_dim]
        batch_size = query.size(0)
        head_dim = self.attention_head_size
        query = query.view(batch_size, -1, self.num_attention_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_attention_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_attention_heads, head_dim).transpose(1, 2)

        # --- TEE处理部分开始 ---
        if self.tee_head_idx is not None:
            # 克隆数据以避免影响原始计算图
            tee_query = query[:, self.tee_head_idx, :, :].clone().detach()
            tee_key = key[:, self.tee_head_idx, :, :].clone().detach()
            tee_value = value[:, self.tee_head_idx, :, :].clone().detach()

            # 准备TEE输入
            tee_input = {
                "query": tee_query,
                "key": tee_key,
                "value": tee_value,
                "layer_name": self.layer_name,
                "head_idx": self.tee_head_idx,
                "attention_mask": attention_mask[:, :, self.tee_head_idx, :].clone().detach()
                if attention_mask is not None else None
            }

            # 保存输入并等待TEE处理
            input_path = os.path.join(gradient_dir, f"tee_input_{self.layer_name}_head{self.tee_head_idx}.pt")
            torch.save(tee_input, input_path)

            # 等待TEE返回结果
            output_path = os.path.join(output_dir, f"tee_output_{self.layer_name}_head{self.tee_head_idx}.pt")
            tee_output = load_with_retry(output_path)
            os.remove(output_path)

            # 用TEE结果替换原始值（注意保留梯度计算）
            with torch.no_grad():
                query[:, self.tee_head_idx, :, :] = tee_output["attention_output"]
                value[:, self.tee_head_idx, :, :] = tee_output["processed_value"]
        # --- TEE处理部分结束 ---

        # 计算注意力分数（保持原始实现）
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(
            torch.tensor(head_dim, dtype=attention_scores.dtype, device=attention_scores.device)
        )

        # 处理attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.expand(-1, self.num_attention_heads, -1, -1)
            attention_scores = attention_scores + attention_mask

        # 使用原始softmax实现（不要改为nn.Softmax！）
        # 将softmax送入TEE
        tee_softmax_input = {
            "hidden_states": attention_scores.detach(),
            "layer_name": f"base_model.model.roberta.encoder.{self.layer_name}.attention.self.softmax",
            "head_idx": self.tee_head_idx,
            "layer_type": "Softmax"
        }
        tee_input_path = os.path.join(gradient_dir, f"tee_input_base_model.model.roberta.encoder.{self.layer_name}.attention.self.softmax,pt")
        torch.save(tee_softmax_input, tee_input_path)

        # 等待tee返回计算结果
        while True:
            tee_output_files = os.listdir(output_dir)
            if f"tee_output_base_model.model.roberta.encoder.{self.layer_name}.attention.self.softmax.pt" in tee_output_files:
                # print(f"tee output for layer {layer_name} received from TEE")
                break

        # 等待TEE返回结果
        tee_output_path = os.path.join(output_dir, f"tee_output_base_model.model.roberta.encoder.{self.layer_name}.attention.self.softmax.pt")
        tee_softmax_output = load_with_retry(tee_output_path)
        os.remove(tee_output_path)
        # 使用桥接
        attention_probs = bridge_softmax(
            tee_softmax_output["encrypted_output"],
            nn.functional.softmax(attention_scores, dim=-1)
        )
        # attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        # 应用head mask（如果有）
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文向量
        context = torch.matmul(attention_probs, value)

        # 合并多头 [batch_size, seq_len, hidden_size]
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.num_attention_heads * head_dim)

        # 输出处理
        outputs = (context, attention_probs) if output_attentions else (context,)
        return outputs
    def backward_hook(self, grad_output, qkv_type):
        """捕获lora梯度"""
        # 确保 cached_input 是张量
        # 根据类型选择正确的缓存输入
        cached_input = None
        if qkv_type == 'query':
            cached_input = self.cached_query_input
        elif qkv_type == 'key':
            cached_input = self.cached_key_input
        elif qkv_type == 'value':
            cached_input = self.cached_value_input
        if isinstance(cached_input, (dict, OrderedDict)):
            # 如果是字典/OrderedDict，尝试获取第一个值
            cached_input = next(iter(cached_input.values()))

        if not isinstance(cached_input, torch.Tensor):
            raise ValueError(f"cached_input 必须是张量，但得到 {type(cached_input)}")
        grad_request = {
            "layer_name": self.layer_name,
            "input": cached_input.detach().clone(),
            "grad_output": grad_output.clone(), # 下一层的输入
            "qkv_type": qkv_type,
            "scaling": getattr(self, "scaling", 1.0)
        }
        print(f"grad_output: {grad_request['grad_output'].norm().item()}")
        # 把梯度传递给TEE
        grad_path = os.path.join(gra_dir, f"grad_request_{self.layer_name}_{qkv_type}.pt")
        torch.save(grad_request, grad_path)
        # print(f"grad_request_{self.layer_name}_{qkv_type}.pt has been saved")


def load_with_retry(file_path, max_retries=600, delay=0.01):
    for _ in range(max_retries):
        try:
            return torch.load(file_path, map_location=torch.device('cuda:1'), weights_only=False)
        except:
            time.sleep(delay)
    raise RuntimeError(f"Failed to load {file_path} after {max_retries} retries")

# 修改模型的 forward 方法，提取 [CLS] 标记的隐藏状态
def custom_forward(*args, **kwargs):
    # 过滤掉 labels 参数
    roberta_kwargs = {k: v for k, v in kwargs.items() if k != "labels"}

    # 调用 roberta 部分的前向传播
    roberta_outputs = model.roberta(*args, **roberta_kwargs)

    # 提取 [CLS] 标记的隐藏状态
    cls_hidden_state = roberta_outputs.last_hidden_state[:, 0, :]  # 形状为 [batch_size, hidden_size]
    # 将 [CLS] 标记的隐藏状态传递给分类头
    print(cls_hidden_state.shape)
    logits = model.classifier(cls_hidden_state)
    # 返回结果
    return {"logits": logits}


# tokenizer_model = AutoTokenizer.from_pretrained("philschmid/roberta-large-sst2")
# 加载分词模型
tokenizer_model = AutoTokenizer.from_pretrained("roberta-large",
                                                cache_dir = "/root/data_dir/LLM_finetune/local_tokenizer_model/models--roberta-large")

# 加载数据集
ds = load_dataset("/root/data_dir/LLM_finetune/data/glue/rte")

# 记载模型
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-large",
    num_labels=2,
    cache_dir="/root/data_dir/",
    torch_dtype="auto",
    device_map="cuda:1",
    low_cpu_mem_usage=True,
    output_hidden_states=True)

# model = AutoModelForSequenceClassification.from_pretrained("philschmid/roberta-large-sst2")
for name, param in model.classifier.named_parameters():
    print(name, param.shape)

print(model)

# 先保存一下原始的forward
original_forward = model.forward
# 替换模型中的自注意力机制
# replace_self_attention(model)

config = LoraConfig(task_type=TaskType.SEQ_CLS,
                    r=8,
                    lora_alpha=32,
                    target_modules=['query', 'key', 'value'],
                    modules_to_save=[],
                    lora_dropout=0.05,
                    bias="none")

peft_model = get_peft_model(model, config).to(device='cuda:1')

def extract_lora_layers(model):
    """提取LORA"""
    lora_params = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            lora_params[name] = module
    return lora_params

lora_layers = extract_lora_layers(peft_model)
print(lora_layers)

def convert_to_tee_model(model: Union[RobertaForSequenceClassification, PeftModel], tee_head_indices: dict):
    """
    将普通Roberta模型转换为支持TEE的版本
    :param model: 原始Roberta模型
    :param tee_head_indices: 字典，指定哪些层的哪些头需要TEE处理
        示例: {"layer.3": 1} 表示第4层的第2个注意力头(从0开始)
    """
    base_model = model if not isinstance(model, PeftModel) else model.base_model
    for layer_idx, layer in enumerate(base_model.roberta.encoder.layer):
        layer_name = f"layer.{layer_idx}"
        # if layer_name in tee_head_indices:
        original = layer.attention.self
        custom = CustomRobertaSelfAttention(base_model.config)
        # 关键步骤：完全复制原始参数

        custom.query.weight.copy_(original.query.base_layer.weight.data)
        custom.query.bias.copy_(original.query.base_layer.bias.data)
        custom.key.weight.copy_(original.key.base_layer.weight.data)
        custom.key.bias.copy_(original.key.base_layer.bias.data)
        custom.value.weight.copy_(original.value.base_layer.weight.data)
        custom.value.bias.copy_(original.value.base_layer.bias.data)

        # 设置TEE相关参数
        # custom.tee_head_idx = tee_head_indices[layer_name] #***
        custom.tee_head_idx = None
        custom.layer_name = layer_name

        # 添加LoRA参数（如果原始层有）
        if hasattr(original, 'query') and hasattr(original.query, 'lora_A'):
            custom.lora_query = original.query
            custom.lora_key = original.key
            custom.lora_value = original.value
            custom.register_lora_hooks()

        # 替换原始层
        layer.attention.self = custom
    return model


# 3. 转换为TEE模型（示例：第0层的第1个头，第3层的第2个头）
tee_config = {
    "layer.0": 1,  # 第1层的第2个头
    "layer.3": 2   # 第4层的第3个头
}

tee_model = convert_to_tee_model(peft_model, tee_config)
tee_model.to(device='cuda:1')
tee_model.forward = types.MethodType(custom_forward, tee_model)
model = tee_model

def get_embed_layer(model):
    """提取embedding层"""
    embedding_layers = {}
    # 先找到全部nn.embedding层，并记录其父级前缀
    embedding_prefixes = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            parent_prefix = ".".join(name.split(".")[:-1]) #获取父级模块
            embedding_prefixes.add(parent_prefix) #记录前缀
    # 遍历所有模块
    for name, module in model.named_modules():
        if any(name.startswith(prefix) for prefix in embedding_prefixes):
            embedding_layers[name] = module
    return embedding_layers

embedding_layers = get_embed_layer(peft_model)

def extract_nonembed_layers(model):
    """提取除了embedding之外的其他层"""
    nonembedding_layers = {}
    for name, module in model.named_modules():
        if name in embedding_layers:
            continue
        else:
            nonembedding_layers[name] = module
    return nonembedding_layers

nonembedding_layers = extract_nonembed_layers(peft_model)

def extract_nonlinear_layers(model):
    """提取非线性层"""
    nonlinear_layers = {}
    # 先获取所有embedding相关的层名
    for name, module in model.named_modules():
        # 先排除embedding相关层
        if name in embedding_layers:
            continue
        if isinstance(module, (nn.ReLU, GELUActivation, nn.LayerNorm, nn.Softmax)):
            nonlinear_layers[name] = {
                "type": module.__class__.__name__,
                "params": {k: v.clone() for k, v in module.state_dict().items()}
            }
    return nonlinear_layers
nonlinear_layers = extract_nonlinear_layers(peft_model)

print("trainable param:")
for name, param in model.named_parameters():
    if "lora" not in name:
        param.requires_grad = False
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.shape)
total_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
print(f"total params: {total_params / 1e6:.2f}M")

class SecureBridge(nn.Module):
    def __init__(self):
        super().__init__()
        # 无任何可训练参数

    def forward(self, tee_output, gpu_input):
        """
        tee_output: 来自TEE的加密计算结果（无梯度）
        gpu_input: GPU端的原始输入（保留梯度）
        """
        # 完全使用TEE数值，但用GPU输入的梯度路径
        return tee_output.detach() + (gpu_input - tee_output.detach())

bridge_ln = SecureBridge().to('cuda:1')    # 用于LayerNorm层
bridge_gelu = SecureBridge().to('cuda:1')  # 用于GELU层
bridge_embed = SecureBridge().to('cuda:1') # 用于embedding层
bridge_softmax = SecureBridge().to('cuda:1') # 用于SOFTMAX层
bridge_lora = SecureBridge().to('cuda:1') # 用于LORA层

# OPT加密函数
def otp_encrypt(data, mask, prime=257):
    return (data + mask)

# OPT解密函数
def otp_decrypt(data, mask, prime=257):
    return (data - mask)

# 量化函数
def quantize(data, bits=8, clip_range=None):
    if clip_range is not None:
        data = torch.clamp(data, clip_range[0], clip_range[1])
    min_val = data.min()
    max_val = data.max()
    scale = (max_val - min_val) / (2**bits - 1)
    quantized_data = ((data - min_val) / scale).round().clamp(0, 2**bits - 1).to(torch.int8)
    return quantized_data, scale, min_val

# 反量化函数
def dequantize(quantized_data, scale, min_val):
    return quantized_data.float() * scale + min_val


gradient_dir = "/root/host_dir/occlum_instance/image/LLM_finetune/sgx_host/shared_folder_robertarte/local_gradients"  # 本地模型传来的文件
output_dir = "/root/host_dir/occlum_instance/image/LLM_finetune/sgx_host/shared_folder_robertarte/lora_gradients"  # lora传来的文件
gra_dir = "/root/host_dir/occlum_instance/image/LLM_finetune/sgx_host/shared_folder_robertarte/gra_gradients"  # 本地模型传来的梯度
gra_upd_dir = "/root/host_dir/occlum_instance/image/LLM_finetune/sgx_host/shared_folder_robertarte/gra_upd_gradients"  # lora更新后的梯度
log_dir = "/root/host_dir/occlum_instance/image/LLM_finetune/sgx_host/shared_folder_robertarte/tensorboard_logs"  # tensorboard
model_dir = "/root/host_dir/occlum_instance/image/LLM_finetune/sgx_host/shared_folder_robertarte/seved_model"
mask_dir = "/root/host_dir/occlum_instance/image/LLM_finetune/sgx_host/shared_folder_robertarte/mask_and_layers"  # 保存掩码和提取出来的各个层

os.makedirs(gra_dir, exist_ok=True)
os.makedirs(gra_upd_dir, exist_ok=True)
os.makedirs(gradient_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

# 将非线性层保存下来传输到SGX
output_file = os.path.join(mask_dir, "nonlinear_layers.pt")
torch.save(nonlinear_layers, output_file)
print(f"Nonlinear layers saved to {output_file}")

# 将LORA层保存下来传输到SGX
output_file = os.path.join(mask_dir, "lora_layers.pt")
torch.save(lora_layers, output_file)
print(f"lora layers saved to {output_file}")
# 处理数据
"""
并将其转换成适合用于模型训练的输入格式。具体来说，
它将原始的输入数据（如用户指令、用户输入、助手输出等）转换为模型所需的格式，
包括 input_ids、attention_mask 和 labels。
"""

def process_func(example):
    example['labels'] = example['label']
    return tokenizer_model(
        example["sentence1"],  # 替换原来的 question1/premise
        example["sentence2"],  # 替换原来的 question2/hypothesis
        truncation=True,
        padding="max_length",
        max_length=128
    )

# 分词
tokenized_ds = ds.map(process_func, batched=True)

# 设置为张量格式
tokenized_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# small_ds = tokenized_ds["train"].train_test_split(train_size=0.1, shuffle=True)["train"]
# train_dataloader = DataLoader(small_ds, batch_size=64, shuffle=True, drop_last=True)
train_dataloader = DataLoader(tokenized_ds["train"], batch_size=64, shuffle=True, drop_last=True)
val_dataloader = DataLoader(tokenized_ds["validation"], batch_size=64, shuffle=True, drop_last=True)
test_dataloader = DataLoader(tokenized_ds["test"], batch_size=64, shuffle=True, drop_last=True)


criterion = nn.CrossEntropyLoss()
# 修改你的优化器初始化
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4)
epoch_loss = 0.0
epoch_acc = 0.0
num_batches = len(train_dataloader)


def train_step(batch):
    inputs = batch["input_ids"].to("cuda:1")
    attention_mask = batch["attention_mask"].to("cuda:1")
    labels = batch["labels"].to("cuda:1")

    hidden_states = model.roberta.embeddings(inputs)
    tee_embed_input = {
        "hidden_states": hidden_states,
        "attention_mask": attention_mask,
        "layer_type": "encrypt"
    }
    tee_input_file = os.path.join(gradient_dir, "tee_input_encrypt.pt")
    torch.save(tee_embed_input, tee_input_file)
    print(f"data sent to tee for encryption")

    # 等待TEE返回加密后的数据
    while True:
        tee_output_files = os.listdir(output_dir)
        if "tee_output_encrypt.pt" in tee_output_files:
            print("encrypted data received")
            break

    # 加载tee返回的加密数据
    tee_output_file = os.path.join(output_dir, "tee_output_encrypt.pt")
    encrypted_tee_output = load_with_retry(tee_output_file)
    os.remove(tee_output_file)

    # 使用桥接
    raw_embed = hidden_states  # 保存原始embedding输出
    tee_output = encrypted_tee_output["encrypted_output"].to(torch.float32)
    hidden_states = bridge_embed(tee_output, raw_embed)

    residual = hidden_states
    extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * torch.finfo(torch.float32).min

    for layer_idx, layer in enumerate(model.roberta.encoder.layer):
        layer_name = f"base_model.model.roberta.encoder.layer.{layer_idx}"
        # Attention部分
        # 保存残差连接输入
        attention_residual = residual
        self_outputs = layer.attention.self(hidden_states, extended_attention_mask)
        attention_output = layer.attention.output.dense(self_outputs[0])
        attention_output = layer.attention.output.dropout(attention_output)

        # hidden_states = layer.attention.output.LayerNorm(attention_output + hidden_states)
        # 将密文数据发送到tee
        tee_input_data = {
            "hidden_states": (attention_output + hidden_states).detach().cpu(),
            "attention_mask": extended_attention_mask,
            "layer_name": f"{layer_name}.attention.output.LayerNorm",
            "layer_type": "LayerNorm"  # 保存层类型信息
        }
        tee_input_file = os.path.join(gradient_dir, f"tee_input_{layer_name}.pt")
        torch.save(tee_input_data, tee_input_file)

        # 等待tee返回计算结果
        while True:
            tee_output_files = os.listdir(output_dir)
            if f"tee_output_{layer_name}.attention.output.LayerNorm.pt" in tee_output_files:
                # print(f"tee output for layer {layer_name} received from TEE")
                break

        # 加载tee返回的加密结果
        tee_output_file = os.path.join(output_dir, f"tee_output_{layer_name}.attention.output.LayerNorm.pt")
        encrypted_tee_output = load_with_retry(tee_output_file)
        os.remove(tee_output_file)
        raw_hidden = layer.attention.output.LayerNorm(attention_output + hidden_states)  # 保存残差结果
        tee_output = encrypted_tee_output["encrypted_output"]
        hidden_states = bridge_ln(tee_output, raw_hidden)  # 梯度桥接

        ffn_residual = hidden_states
        # FFN部分
        intermediate_output = layer.intermediate.dense(hidden_states)
        # intermediate_output = layer.intermediate.intermediate_act_fn(intermediate_output)
        # 将密文数据发送到tee
        tee_input_data = {
            "hidden_states": intermediate_output,
            "attention_mask": extended_attention_mask,
            "layer_name": f"{layer_name}.intermediate.intermediate_act_fn",
            "layer_type": "GELUActivation"  # 保存层类型信息
        }
        tee_input_file = os.path.join(gradient_dir, f"tee_input_{layer_name}.pt")
        torch.save(tee_input_data, tee_input_file)
        # print(f"data for layer {layer_name} sent to TEE")

        # 等待tee返回计算结果
        while True:
            tee_output_files = os.listdir(output_dir)
            if f"tee_output_{layer_name}.intermediate.intermediate_act_fn.pt" in tee_output_files:
                # print(f"tee output for layer {layer_name} received from TEE")
                break

        # 加载tee返回的加密结果
        tee_output_file = os.path.join(output_dir, f"tee_output_{layer_name}.intermediate.intermediate_act_fn.pt")
        encrypted_tee_output = load_with_retry(tee_output_file)
        os.remove(tee_output_file)

        # 更新intermediate_output
        # intermediate_output = encrypted_tee_output["encrypted_output"].to(torch.float32)
        raw_intermediate = layer.intermediate.intermediate_act_fn(intermediate_output)  # 保存原始输入
        tee_output = encrypted_tee_output["encrypted_output"].to(torch.float32)
        # print("tee_output", tee_output)
        intermediate_output = bridge_gelu(tee_output, raw_intermediate)  # 梯度桥接

        ffn_output = layer.output.dense(intermediate_output)
        ffn_output = layer.output.dropout(ffn_output)
        hidden_states1 = layer.output.LayerNorm(ffn_output + hidden_states)
        # # 将密文数据发送到tee
        tee_input_data = {
            "hidden_states": ffn_output + ffn_residual,
            "attention_mask": extended_attention_mask,
            "layer_name": f"{layer_name}.output.LayerNorm",
            "layer_type": "LayerNorm"  # 保存层类型信息
        }
        tee_input_file = os.path.join(gradient_dir, f"tee_input_{layer_name}.pt")
        torch.save(tee_input_data, tee_input_file)
        # print(f"data for layer {layer_name} sent to TEE")

        # 等待tee返回计算结果
        while True:
            tee_output_files = os.listdir(output_dir)
            if f"tee_output_{layer_name}.output.LayerNorm.pt" in tee_output_files:
                # print(f"tee output for layer {layer_name} received from TEE")
                break
        # 加载tee返回的加密结果
        tee_output_file = os.path.join(output_dir, f"tee_output_{layer_name}.output.LayerNorm.pt")
        encrypted_tee_output = load_with_retry(tee_output_file)
        os.remove(tee_output_file)
        # hidden_states = encrypted_tee_output["encrypted_output"].to(torch.float32)
        raw_hidden = layer.output.LayerNorm(ffn_output + ffn_residual)  # 保存残差结果
        tee_output = encrypted_tee_output["encrypted_output"].to(torch.float32)
        hidden_states = bridge_ln(tee_output, raw_hidden)  # 梯度桥接

    # 分类头
    logits = model.classifier(hidden_states)  # 取[CLS] token

    # 计算loss
    loss = criterion(logits, labels)
    accuracy = (logits.argmax(-1) == labels).float().mean()
    print("loss", loss)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    # for name, param in model.named_parameters():
    #     if 'lora_' in name and param.grad is not None:
    #         print(f"{name} grad norm: {param.grad.norm().item()}")
    optimizer.step()

    return loss.item(), accuracy.item()


def evaluate(model, dataloader):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input_ids"].to("cuda:1")
            attention_mask = batch["attention_mask"].to("cuda:1")
            labels = batch["labels"].to("cuda:1")

            hidden_states = model.roberta.embeddings(inputs)
            tee_embed_input = {
                "hidden_states": hidden_states,
                "attention_mask": attention_mask,
                "layer_type": "encrypt"
            }
            tee_input_file = os.path.join(gradient_dir, "tee_input_encrypt.pt")
            torch.save(tee_embed_input, tee_input_file)
            print(f"data sent to tee for encryption")

            # 等待TEE返回加密后的数据
            while True:
                tee_output_files = os.listdir(output_dir)
                if "tee_output_encrypt.pt" in tee_output_files:
                    print("encrypted data received")
                    break

            # 加载tee返回的加密数据
            tee_output_file = os.path.join(output_dir, "tee_output_encrypt.pt")
            encrypted_tee_output = load_with_retry(tee_output_file)
            os.remove(tee_output_file)

            # 使用桥接
            raw_embed = hidden_states  # 保存原始embedding输出
            tee_output = encrypted_tee_output["encrypted_output"].to(torch.float32)
            hidden_states = bridge_embed(tee_output, raw_embed)

            residual = hidden_states
            extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * torch.finfo(torch.float32).min

            for layer_idx, layer in enumerate(model.roberta.encoder.layer):
                layer_name = f"base_model.model.roberta.encoder.layer.{layer_idx}"
                # Attention部分
                # 保存残差连接输入
                attention_residual = residual
                self_outputs = layer.attention.self(hidden_states, extended_attention_mask)
                attention_output = layer.attention.output.dense(self_outputs[0])
                attention_output = layer.attention.output.dropout(attention_output)

                hidden_states1 = layer.attention.output.LayerNorm(attention_output + hidden_states)
                # 将密文数据发送到tee
                tee_input_data = {
                    "hidden_states": attention_output + hidden_states,
                    "attention_mask": extended_attention_mask,
                    "layer_name": f"{layer_name}.attention.output.LayerNorm",
                    "layer_type": "LayerNorm"  # 保存层类型信息
                }
                tee_input_file = os.path.join(gradient_dir, f"tee_input_{layer_name}.pt")
                torch.save(tee_input_data, tee_input_file)

                # 等待tee返回计算结果
                while True:
                    tee_output_files = os.listdir(output_dir)
                    if f"tee_output_{layer_name}.attention.output.LayerNorm.pt" in tee_output_files:
                        # print(f"tee output for layer {layer_name} received from TEE")
                        break

                # 加载tee返回的加密结果
                tee_output_file = os.path.join(output_dir, f"tee_output_{layer_name}.attention.output.LayerNorm.pt")
                encrypted_tee_output = load_with_retry(tee_output_file)
                os.remove(tee_output_file)
                raw_hidden = layer.attention.output.LayerNorm(attention_output + hidden_states)  # 保存残差结果
                tee_output = encrypted_tee_output["encrypted_output"]
                hidden_states = bridge_ln(tee_output, raw_hidden)  # 梯度桥接

                ffn_residual = hidden_states
                # FFN部分
                intermediate_output = layer.intermediate.dense(hidden_states)
                intermediate_output1 = layer.intermediate.intermediate_act_fn(intermediate_output)
                # 将密文数据发送到tee
                tee_input_data = {
                    "hidden_states": intermediate_output,
                    "attention_mask": extended_attention_mask,
                    "layer_name": f"{layer_name}.intermediate.intermediate_act_fn",
                    "layer_type": "GELUActivation"  # 保存层类型信息
                }
                tee_input_file = os.path.join(gradient_dir, f"tee_input_{layer_name}.pt")
                torch.save(tee_input_data, tee_input_file)
                # print(f"data for layer {layer_name} sent to TEE")

                # 等待tee返回计算结果
                while True:
                    tee_output_files = os.listdir(output_dir)
                    if f"tee_output_{layer_name}.intermediate.intermediate_act_fn.pt" in tee_output_files:
                        # print(f"tee output for layer {layer_name} received from TEE")
                        break

                # 加载tee返回的加密结果
                tee_output_file = os.path.join(output_dir,
                                               f"tee_output_{layer_name}.intermediate.intermediate_act_fn.pt")
                encrypted_tee_output = load_with_retry(tee_output_file)
                os.remove(tee_output_file)

                # 更新intermediate_output
                # intermediate_output = encrypted_tee_output["encrypted_output"].to(torch.float32)
                raw_intermediate = layer.intermediate.intermediate_act_fn(intermediate_output)  # 保存原始输入
                tee_output = encrypted_tee_output["encrypted_output"].to(torch.float32)
                # print("tee_output", tee_output)
                intermediate_output = bridge_gelu(tee_output, raw_intermediate)  # 梯度桥接

                ffn_output = layer.output.dense(intermediate_output)
                ffn_output = layer.output.dropout(ffn_output)
                hidden_states1 = layer.output.LayerNorm(ffn_output + hidden_states)
                # 将密文数据发送到tee
                tee_input_data = {
                    "hidden_states": ffn_output + ffn_residual,
                    "attention_mask": extended_attention_mask,
                    "layer_name": f"{layer_name}.output.LayerNorm",
                    "layer_type": "LayerNorm"  # 保存层类型信息
                }
                tee_input_file = os.path.join(gradient_dir, f"tee_input_{layer_name}.pt")
                torch.save(tee_input_data, tee_input_file)
                # print(f"data for layer {layer_name} sent to TEE")

                # 等待tee返回计算结果
                while True:
                    tee_output_files = os.listdir(output_dir)
                    if f"tee_output_{layer_name}.output.LayerNorm.pt" in tee_output_files:
                        # print(f"tee output for layer {layer_name} received from TEE")
                        break
                # 加载tee返回的加密结果
                tee_output_file = os.path.join(output_dir, f"tee_output_{layer_name}.output.LayerNorm.pt")
                encrypted_tee_output = load_with_retry(tee_output_file)
                os.remove(tee_output_file)
                # hidden_states = encrypted_tee_output["encrypted_output"].to(torch.float32)
                raw_hidden = layer.output.LayerNorm(ffn_output + ffn_residual)  # 保存残差结果
                tee_output = encrypted_tee_output["encrypted_output"].to(torch.float32)
                hidden_states = bridge_ln(tee_output, raw_hidden)  # 梯度桥接

            # 分类头
            logits = model.classifier(hidden_states)  # 取[CLS] token
            # print(f"logits: {logits.argmax(-1)}, labels: {labels}")
            # 计算loss和accuracy
            if torch.any(labels != -1):
                loss = criterion(logits, labels)
                accuracy = (logits.argmax(-1) == labels).float().mean()

                total_loss += loss.item()
                total_acc += accuracy.item()

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    return avg_loss, avg_acc


if __name__ == "__main__":
    global_step = 0
    best_val_acc = 0
    # 循环训练

    for epoch in range(20):
        print(f"epoch {epoch + 1}")
        # 替换模型的 forward 方法
        model.forward = custom_forward
        print(tqdm(train_dataloader))
        epoch_loss = 0
        epoch_acc = 0
        for batch in tqdm(train_dataloader):
            loss, accuracy = train_step(batch)
            epoch_loss += loss
            epoch_acc += accuracy
            global_step += 1
        epoch_loss /= num_batches
        epoch_acc /= num_batches
        print("num_batches", num_batches)
        print(f"epoch {epoch + 1}, train loss: {epoch_loss}, train acc: {epoch_acc}")
        # 将他们写入tensorboard
        writer.add_scalar("loss", epoch_loss, global_step)
        writer.add_scalar("accuracy", epoch_acc, global_step)

        # 验证阶段

        # model.forward = original_forward
        val_loss, val_acc = evaluate(model, val_dataloader)
        print(f"Epoch {epoch + 1}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        writer.add_scalar("Val/Loss", val_loss, global_step)
        writer.add_scalar("Val/Accuracy", val_acc, global_step)

        best_model_path = os.path.join(model_dir, "best_model")  # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 保存完整模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'accuracy': val_acc,
                'bridge_gelu_state': bridge_gelu.state_dict(),
                'bridge_ln_state': bridge_ln.state_dict(),
                'bridge_softmax_state': bridge_softmax.state_dict(),
                'bridge_embed_state': bridge_embed.state_dict()
            }, best_model_path)
            print(f"New best model saved with Val Acc: {best_val_acc:.4f}")
        fin_model_path = os.path.join(model_dir, "fin_model")  # 保存最佳模型
        # 保存完整模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'accuracy': val_acc,
            'bridge_gelu_state': bridge_gelu.state_dict(),
            'bridge_ln_state': bridge_ln.state_dict(),
            'bridge_softmax_state': bridge_softmax.state_dict(),
            'bridge_embed_state': bridge_embed.state_dict()
        }, fin_model_path)
        print(f"final model saved with Val Acc: {val_acc:.4f}")

    # 训练结束后加载最佳模型
    best_model_path = os.path.join(model_dir, "best_model")  # 保存最佳模型
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    bridge_gelu.load_state_dict(checkpoint['bridge_gelu_state'])
    bridge_ln.load_state_dict(checkpoint['bridge_ln_state'])
    bridge_softmax.load_state_dict(checkpoint['bridge_softmax_state'])
    bridge_embed.load_state_dict(checkpoint['bridge_embed_state'])
    bridge_lora.load_state_dict(checkpoint['bridge_lora_state'])

    # 测试最佳模型
    test_loss, test_acc = evaluate(model, val_dataloader)
    print(f"Final Test Performance - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    writer.add_scalar("Test/Loss", test_loss, global_step)
    writer.add_scalar("Test/Accuracy", test_acc, global_step)

    writer.close()