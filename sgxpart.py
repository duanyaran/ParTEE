import psutil
from networkx.algorithms.bridges import bridges
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig
import os
import torch
import torch.nn as nn
import tensorflow as tf
import gc
torch.set_num_threads(1)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_DATASET_CACHE"] = "/host/image/tmp/hf_cache"
os.makedirs(os.environ["HF_DATASET_CACHE"], exist_ok=True)
print("HF_DATASETS_CACHE:", os.environ.get("HF_DATASETS_CACHE"))
import logging
logging.basicConfig(level=logging.INFO)
tf.get_logger().setLevel('ERROR')
from datasets import load_dataset
from peft import PromptTuningConfig, TaskType, PromptTuningInit, get_peft_model, PeftModel, PromptEncoderConfig, \
    PromptEncoderReparameterizationType, PrefixTuningConfig, LoraConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, TrainingArguments, \
    DataCollatorForSeq2Seq, Trainer, PreTrainedModel
import numpy as np
import time
import shutil
import tempfile
from transformers.modeling_outputs import SequenceClassifierOutput

def load_with_retry(file_path, max_retries=60, delay=0.01):
    for _ in range(max_retries):
        try:
            with open(file_path, 'rb') as f:
                data = torch.load(f, map_location=torch.device('cpu'), weights_only=False)
            return data
        except:
            time.sleep(delay)
    raise RuntimeError(f"Failed to load {file_path} after {max_retries} retries")

def hard_cleanup():
    """强制清理所有PyTorch和系统资源"""
    torch.cuda.empty_cache()
    gc.collect()
    # 释放所有残留张量
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            obj.detach().cpu()
            del obj
    # 清空Python对象缓存
    import sys
    sys._clear_type_cache()

# OPT加密函数
def otp_encrypt(data, mask, prime=257):

    scale = (data.max() - data.min()) / 255
    zero_point = data.min()
    quantized = ((data - zero_point) / scale).round().clamp(0, 255).to(torch.uint8)

    encrypted = ((quantized.int() + mask.int()) % 256).to(torch.uint8)  # 关键修改：加法+模256

    encrypted_float = encrypted.float() * scale + zero_point

    dequantized = quantized.float() * scale + zero_point

    return encrypted


# OPT解密函数
def otp_decrypt(encrypted_float, otp_key, scale, zero_point):

    encrypted_quant = ((encrypted_float - zero_point) / scale).round().to(torch.uint8)

    decrypted_quant = ((encrypted_quant.int() - otp_key.int()) % 256).to(torch.uint8)

    decrypted_float = decrypted_quant.float() * scale + zero_point

    return decrypted_float

# 量化函数
def quantize(data, bits = 8, manual_range=None):
    if manual_range is not None:
        min_val, max_val = manual_range
    else:
        min_val = data.min()
        max_val = data.max()
    scale = (max_val - min_val)/(2**bits -1)
    quantized_data = ((data -min_val) / scale).to(torch.int8)
    return quantized_data, scale, min_val

# 反量化函数
def dequantize(quantized_data, scale, min_val):
    return quantized_data.float() * scale + min_val


def cal_accuracy(outputs, labels):
    # 获取预测的类别
    predictions = torch.argmax(outputs, dim=-1)
    # 忽略标签为-100的位置
    mask = labels != 100
    correct = (predictions[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    accuracy = correct / total if total > 0 else 0
    return accuracy

class LoRaModel(torch.nn.Module):
    def __init__(self, rank = 8):
        super(LoRaModel, self).__init__()
        self.rank = rank
        self.lora_A = nn.Linear(1024, rank)
        self.lora_B = nn.Linear(rank, 1024)
        # 启用参数的梯度
        for param in self.parameters():
            param.requires_grad_(True)
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A.weight, a = np.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.lora_B(self.lora_A(x))

lora_model = LoRaModel().to(torch.device('cpu'))

# model_name = "/host/image/LLM_finetune/data_dir/roberta-large" #***
model_name = "/root/data_dir/roberta-large"
print(f"Loading ds from {model_name}")

config = AutoConfig.from_pretrained(model_name, num_labels=2)
base_model = AutoModelForSequenceClassification.from_config(config=config)
criterion = nn.CrossEntropyLoss()


# 修改模型的 forward 方法，提取 [CLS] 标记的隐藏状态
def custom_forward(*args, **kwargs):
    # 过滤掉 labels 参数
    roberta_kwargs = {k: v for k, v in kwargs.items() if k != "labels"}

    # 调用 roberta 部分的前向传播
    roberta_outputs = base_model.roberta(*args, **roberta_kwargs)

    # 提取 [CLS] 标记的隐藏状态
    cls_hidden_state = roberta_outputs.last_hidden_state[:, 0, :]  # 形状为 [batch_size, hidden_size]
    # 将 [CLS] 标记的隐藏状态传递给分类头
    print(cls_hidden_state.shape)
    logits = base_model.classifier(cls_hidden_state)
    # 返回结果
    return {"logits": logits}

# lora
config = LoraConfig(task_type=TaskType.SEQ_CLS,
                    r=8,
                    lora_alpha=32,
                    target_modules= ['query', 'key', 'value'],
                    lora_dropout=0.05,
                    bias="none",
                    modules_to_save=None)

peft_model = get_peft_model(base_model, config)

print("trainable param:")
for name, param in peft_model.named_parameters():
    if param.requires_grad:
        print(name)

total_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
print(f"total params: {total_params/1e6:.2f}M")


def apply_nonlinear_layer(layer_info, input_tensor):
    """应用存储的非线性层"""
    layer_type = layer_info["type"]
    params = layer_info["params"]

    # 重建层
    if layer_type == "LayerNorm":
        layer = nn.LayerNorm(params["weight"].shape[0])
    elif layer_type == "GELUActivation":
        layer = nn.GELU()
    elif layer_type == "ReLU":
        layer = nn.ReLU()
    elif layer_type == "Softmax":
        layer = nn.Softmax(dim=-1)
    else:
        raise ValueError(f"Unsupported layer type: {layer_type}")

    # 加载参数
    layer.load_state_dict(params)
    return layer(input_tensor)

gradient_dir = "/root/host_dir/occlum_instance/image/LLM_finetune/sgx_host/shared_folder_robertarte/local_gradients" # 本地模型传来的梯度 ***/root/host_dir/occlum_instance/image
output_dir = "/root/host_dir/occlum_instance/image/LLM_finetune/sgx_host/shared_folder_robertarte/lora_gradients" # lora更新后的梯度 ***/host/image
mask_dir = "/root/host_dir/occlum_instance/image/LLM_finetune/sgx_host/shared_folder_robertasst2/mask_and_layers" # lora更新后的梯度 ***
gra_dir = "/root/host_dir/occlum_instance/image/LLM_finetune/sgx_host/shared_folder_robertarte/gra_gradients"  # 本地模型传来的梯度
gra_upd_dir = "/root/host_dir/occlum_instance/image/LLM_finetune/sgx_host/shared_folder_robertarte/gra_upd_gradients"  # lora更新后的梯度
os.makedirs(gra_dir, exist_ok=True)
os.makedirs(gra_upd_dir, exist_ok=True)
os.makedirs(gradient_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-4)

# 获取非线性层
nonlinear_layer = os.path.join(mask_dir, f"nonlinear_layers.pt")
nonlinear_layers = torch.load(nonlinear_layer, map_location=torch.device('cpu'))
# 获取lora层
lora_layer = os.path.join(mask_dir, f"lora_layers.pt")
lora_layers = torch.load(lora_layer, map_location=torch.device('cpu'), weights_only=False)
print(lora_layers)

class TrainableLoRALayer(nn.Module):
    def __init__(self, lora_layer_dict):
        super().__init__()
        # self.lora_A = nn.Parameter(lora_A.clone().detach().requires_grad_(True))
        # self.lora_B = nn.Parameter(lora_B.clone().detach().requires_grad_(True))
        # # 确保 scaling 是 float/Tensor，而不是 dict
        # self.scaling = 4.0
        # self.lora_dropout = nn.Dropout(p=0.05, inplace=False)

        # 从字典加载原始LoRA配置
        self.lora_dropout = nn.Dropout(
            p=lora_layer_dict.lora_dropout.default.p,
            inplace=lora_layer_dict.lora_dropout.default.inplace
        )

        # 初始化参数
        self.lora_A = nn.Parameter(lora_layer_dict.lora_A.default.weight.clone())
        self.lora_B = nn.Parameter(lora_layer_dict.lora_B.default.weight.clone())
        self.scaling = lora_layer_dict.scaling['default']

        print(f"Initialized LoRA layer | "
              f"A shape: {self.lora_A.detach().norm().item()} | "
              f"B shape: {self.lora_B.detach().norm().item()} | "
              f"Scaling: {self.scaling}")

        # 重新初始化（重要！）
        # nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        # nn.init.kaiming_uniform_(self.lora_B, a=np.sqrt(5))

    def forward(self, x):
        dropped = self.lora_dropout(x)
        # lora_A = torch.matmul(dropped, self.lora_A.t())  # shape: [batch, seq, r]
        # lora_B = torch.matmul(lora_A, self.lora_B.t())  # shape: [batch, seq, out_dim]
        print(f"self.lora_A.weight:{self.lora_A.detach().norm().item()}")
        lora_A = torch.nn.functional.linear(dropped, self.lora_A)
        lora_B = torch.nn.functional.linear(lora_A, self.lora_B)

        return lora_B * self.scaling

def low_mem_usage():
    mem_info = psutil.virtual_memory()
    print(f"Memory usage: {mem_info.percent}%")

def process_backward(gradients):
    '''处理反向传播，更新LORA参数'''
    optimizer.zero_grad()

    # 模拟前向传播以构建计算图
    dummy_input = torch.randn(1, 1024, requires_grad=True)
    dummy_output = lora_model(dummy_input)

    # 注入接收到的梯度
    dummy_output.backward(gradient=gradients)

    # 更新参数
    optimizer.step()

    # 返回更新后的参数
    updated_params = {
        name: param.detach().clone()
        for name, param in lora_model.named_parameters()
    }
    return updated_params

def update_lora():
    """TEE内逐层计算非线性部分或lora线性部分"""
    # 初始化AdamW状态存储
    time.sleep(0.01)
    updated_lora_layers = {}
    layer_optimizers = {}
    trainable_layer = {}

    for layer_name, layer in lora_layers.items():
        # 封装成可训练层
        # trainable_layer = TrainableLoRALayer(
        #     lora_A=layer.lora_A.default.weight,
        #     lora_B=layer.lora_B.default.weight,
        #     scaling=getattr(layer, 'scaling', 4.0)
        # )
        print(layer_name)
        trainable_layer[layer_name] = TrainableLoRALayer(layer)
        updated_lora_layers[layer_name] = trainable_layer[layer_name]
        # # 为每层单独创建优化器（确保参数独立更新）
        # layer_optimizers[layer_name] = torch.optim.AdamW(
        #     trainable_layer.parameters(),
        #     lr=1e-4,
        #  )
    # 初始化优化器
    all_params = []
    for layer in trainable_layer.values():
        all_params.extend(list(layer.parameters()))
    optimizer = torch.optim.AdamW(all_params, lr=1e-4)

    global_step = 0
    while True:
        global_step += 1
        tee_input_files = os.listdir(gradient_dir)
        # 优先处理LoRA梯度更新请求
        # 处理梯度更新请求
        # 查找梯度请求文件
        grad_files = [f for f in os.listdir(gra_dir)
                      if f.startswith('grad_request_')]
        if len(grad_files) == 72:
            for grad_file in grad_files:
                try:
                    # 1. 加载请求数据
                    grad_path = os.path.join(gra_dir, grad_file)
                    request = load_with_retry(grad_path)
                    h = request['input'].clone().requires_grad_(True)  # 前向输入 [batch, seq, dim]
                    grad_out = request['grad_output']  # 下一层梯度 [batch, seq, dim]
                    scaling = request['scaling']
                    qkv_type = request['qkv_type']
                    layer_name = f"base_model.model.roberta.encoder.{request['layer_name']}.attention.self.{qkv_type}"
                    # 2. 获取对应LoRA层
                    trainable_layer = updated_lora_layers[layer_name]
                    # optimizers = layer_optimizers[layer_name]

                    optimizer.zero_grad()
                    # 前传
                    output = trainable_layer(h)
                    # 反传
                    output.backward(grad_out)
                    optimizer.step()
                    print(f"LoRA output norm: {output.norm().item()}")
                    # print(f"lora_A.grad.norm(): {trainable_layer.lora_A.grad.norm().item()}")
                    # print(f"lora_B.grad.norm(): {trainable_layer.lora_B.grad.norm().item()}")
                    # print(
                    #     f"trainable_layer.lora_A.detach().norm().item() {trainable_layer.lora_A.detach().norm().item()},"
                    #     f"trainable_layer.lora_B.detach().norm().item() {trainable_layer.lora_B.detach().norm().item()}")
                    print(f"h norm: {h.norm()}")
                    print(f"grad_out norm: {grad_out.norm()}")
                    print(f"LoRA output norm: {output.norm().item()}")
                    # print(f"lora_A.grad.norm(): {trainable_layer.lora_A.grad.norm().item()}")
                    # print(f"lora_B.grad.norm(): {trainable_layer.lora_B.grad.norm().item()}")
                    # print(f"trainable_layer.lora_A.detach().norm().item() {trainable_layer.lora_A.detach().norm().item()},"
                    #       f"trainable_layer.lora_B.detach().norm().item() {trainable_layer.lora_B.detach().norm().item()}")
                    # 6. 保存更新后的参数
                    updated_params = {
                        f"{layer_name}.lora_A.default.weight": trainable_layer.lora_A.detach(),
                        f"{layer_name}.lora_B.default.weight": trainable_layer.lora_B.detach()
                    }
                    torch.save(updated_params, os.path.join(gra_upd_dir, f'updated_lora_params_{request["layer_name"]}_{qkv_type}.pt'))

                finally:
                    os.remove(os.path.join(gra_dir, grad_file))
                    hard_cleanup()
                    del grad_path, request, h, grad_out, scaling, qkv_type
                    print(f"deleted grad file: {grad_file}")
        elif len(grad_files) > 0:
            time.sleep(1)
        elif not os.listdir(gra_dir):
            if any("tee_input" in f for f in tee_input_files):
                # 加载数据
                # tee_input_file = [os.path.join(gradient_dir, f) for f in tee_input_files if f.startswith("tee_input_base_model.model.roberta.encoder.layer.0.out")]
                tee_input_file = os.path.join(gradient_dir, tee_input_files[0])
                tee_input_data = load_with_retry(tee_input_file)
                os.remove(tee_input_file)
                print(f"data received from layer {tee_input_file}")

                # 提取数据
                hidden_states = tee_input_data.get("hidden_states", None)
                attention_mask = tee_input_data.get("attention_mask", None)
                layer_name = tee_input_data.get("layer_name", "unknown")
                layer_type = tee_input_data.get("layer_type", "linear")
                original_proj = tee_input_data.get("original_proj", None)
                scale = tee_input_data.get("scale", 1.0)
                min_val = tee_input_data.get("min_val", 0.0)
                labels = tee_input_data.get("labels", None)
                print(f"layer_type:{layer_type}")

                if layer_type == "encrypt": # 加密操作
                    # 在tee内进行量化和加密
                    # quantized_output, scale, min_val = quantize(hidden_states) # ***
                    quantized_output = hidden_states
                    # 从掩码字典中找出对应层的名字

                    mask = torch.load(f"{mask_dir}/r_encrypt.pt", map_location=torch.device('cpu'))
                    encrypted_output = otp_encrypt(quantized_output, mask)

                    # 保存加密后的结果
                    encrypted_result = {
                        "encrypted_output": encrypted_output,
                        "scale": scale,
                        "min_val": min_val
                    }
                    tee_output_file = os.path.join(output_dir, "tee_output_encrypt.pt")
                    torch.save(encrypted_result, tee_output_file)
                    print(f"encrypted data saved to {tee_output_file}")
                elif layer_type in ["ReLU", "GELUActivation", "Softmax", "LayerNorm"]:

                    # 在tee内进行解密
                    # 从hr字典中找出对应层的名字
                    with torch.no_grad():
                        mask = torch.load(f"{mask_dir}/hr_{layer_name}.pt", map_location=torch.device('cpu'))
                        decrypted_output = otp_decrypt(hidden_states, mask) #解密
                        # 反量化
                        # dequantized_output = dequantize(decrypted_output, scale, min_val) # ***
                        dequantized_output = decrypted_output

                    # 根据层类型进行计算
                    if layer_type == "ReLU":
                        nonlinear_output = torch.relu(dequantized_output)
                        # 仍然在tee内进行量化和加密
                        # quantized_output, scale, min_val = quantize(nonlinear_output) #****
                        quantized_output = nonlinear_output
                    elif layer_type == "GELUActivation":
                        # nonlinear_output = torch.nn.functional.gelu(dequantized_output)
                        nonlinear_output = apply_nonlinear_layer(nonlinear_layers[layer_name], dequantized_output)
                        # 仍然在tee内进行量化和加密
                        # quantized_output, scale, min_val = quantize(nonlinear_output) #****
                        quantized_output = nonlinear_output
                    elif layer_type == "LayerNorm":
                        nonlinear_output = apply_nonlinear_layer(nonlinear_layers[layer_name], dequantized_output)
                        # 仍然在tee内进行量化和加密
                        # quantized_output, scale, min_val = quantize(nonlinear_output) #****
                        quantized_output = nonlinear_output
                    elif layer_type == "Softmax":
                        nonlinear_output = nn.functional.softmax(dequantized_output, dim=-1)
                        quantized_output = nonlinear_output

                    mask = torch.load(f"{mask_dir}/r_{layer_name}.pt", map_location=torch.device('cpu'))
                    if "layer.23.output.L" in layer_name:
                        mask = 0
                    encrypted_output = otp_encrypt(quantized_output, mask)
                    print(encrypted_output.requires_grad)
                    # 保存加密后的结果
                    encrypted_result = {
                        "encrypted_output": encrypted_output,
                        "scale": scale,
                        "min_val": min_val
                    }

                    tee_output_file = os.path.join(output_dir, f"tee_output_{layer_name}.pt")
                    torch.save(encrypted_result, tee_output_file)
                    print(f"encrypted data saved to {tee_output_file}")
                elif layer_type == "lora":
                    with torch.no_grad():
                        mask = torch.load(f"{mask_dir}/hr_{layer_name}.pt", map_location=torch.device('cpu'))
                        decrypted_output = otp_decrypt(hidden_states, mask) #解密
                        # 反量化
                        # dequantized_output = dequantize(decrypted_output, scale, min_val) # ***
                        dequantized_output = decrypted_output

                        # 使用训练后的新参数
                        print("layername", layer_name)
                        trainable_layer = updated_lora_layers[layer_name]
                        nonlinear_output = trainable_layer(dequantized_output)
                        # print(
                        #     f"trainable_layer.lora_A.detach().norm().item() {trainable_layer.lora_A.detach().norm().item()},"
                        #     f"trainable_layer.lora_B.detach().norm().item() {trainable_layer.lora_B.detach().norm().item()}")

                        quantized_output = nonlinear_output
                        layer_name = f"{layer_name}.lora"
                        mask = torch.load(f"{mask_dir}/r_{layer_name}.pt", map_location=torch.device('cpu'))
                        if "layer.23.output.L" in layer_name:
                            mask = 0
                        encrypted_output = otp_encrypt(quantized_output, mask)
                        print(encrypted_output.requires_grad)
                        # 保存加密后的结果
                        encrypted_result = {
                            "encrypted_output": encrypted_output,
                            "scale": scale,
                            "min_val": min_val
                        }

                        tee_output_file = os.path.join(output_dir, f"tee_output_{layer_name}.pt")
                        torch.save(encrypted_result, tee_output_file)
                        print(f"encrypted data saved to {tee_output_file}")

        try:
            # 删除所有可能的临时变量
            del (
                tee_input_files, tee_input_data, hidden_states, quantized_output, mask, min_val,
                encrypted_output, encrypted_result, tee_output_file, decrypted_output,
                dequantized_output, nonlinear_output, grad_files, grad_file,
                grad_path, request, h, grad_out, scaling, qkv_type, layer_name, layer,
                updated_params, layer_type, original_proj, labels, nonlinear_layer, lora_layer,
                global_step, scaling
            )
            gc.collect()
            # 额外清理可能存在的其他变量
            if 'nonlinear_output' in locals(): del nonlinear_output
            if 'grad_out' in locals(): del grad_out
            if 'grad_A' in locals(): del grad_A
            if 'grad_B' in locals(): del grad_B
        except (UnboundLocalError, NameError) as e:
            # 忽略变量不存在的错误
            pass

if __name__ == "__main__":
    update_lora()
