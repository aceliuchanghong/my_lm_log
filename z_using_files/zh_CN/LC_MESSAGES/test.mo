��          �   %   �      `  	   a  �   k          (     0     =     C  x   I     �     �     �     �  C   �  /   7  _   g    �     �     �  �      V   �  V     O   u     �  N   �        N   ,  N  {  	   �	  �   �	     �
  	   �
     �
     �
  	   �
  �   �
     P     c     j     w  8   �     �  F   �  �   )  T   �       �   1  5   �  S   0  N   �  	   �  M   �  	   +  M   5                                   
                                                                               	          # Prompts For vLLM, the memory usage is not reported because it pre-allocates all GPU memory. We use ``gpu_memory_utilization=0.9 max_model_len=32768 enforce_eager=False`` by default. GPU Memory(GB) GPU Num Input Length Model Note: Note: Compared with dense models, MOE models have larger throughput when batch size is large, which is shown as follows: QPS Quantization Speed Benchmark Speed(tokens/s) The environment of the evaluation with huggingface transformers is: The environment of the evaluation with vLLM is: The results are obtained from vLLM throughput benchmarking scripts, which can be reproduced by: This section reports the speed performance of bf16 models, quantized models (including GPTQ-Int4, GPTQ-Int8 and AWQ) of the Qwen2 series. Specifically, we report the inference speed (tokens/s) as well as memory footprint (GB) under the conditions of different context lengths. To be updated for Qwen2.5. Tokens/s We test the speed and memory of generating 2048 tokens with the input lengths of 1, 6144, 14336, 30720, 63488, and 129024 tokens (\>32k is only avaliable for Qwen2-72B-Instuct and Qwen2-7B-Instuct). We use the batch size of 1 and the least number of GPUs as possible for the evalution. [Default Setting]=(gpu_memory_utilization=0.9 max_model_len=32768 enforce_eager=False) [Setting 1]=(gpu_memory_utilization=0.98 max_model_len=4096 enforce_eager=True) [Setting 2] [Setting 2]=(gpu_memory_utilization=1.0 max_model_len=4096 enforce_eager=True) [Setting 3] [Setting 3]=(gpu_memory_utilization=1.0 max_model_len=8192 enforce_eager=True) Project-Id-Version: Qwen 
Report-Msgid-Bugs-To: 
PO-Revision-Date: YEAR-MO-DA HO:MI+ZONE
Last-Translator: FULL NAME <EMAIL@ADDRESS>
Language: zh_CN
Language-Team: zh_CN <LL@li.org>
Plural-Forms: nplurals=1; plural=0;
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.15.0
 请求数 对于vLLM，由于GPU显存预分配，实际显存使用难以评估。默认情况下，统一设定为``gpu_memory_utilization=0.9 max_model_len=32768 enforce_eager=False``。 显存占用 (GB) GPU数量 输入长度 模型 注意： 混合专家模型 (Mixture-of-Experts, MoE) 与稠密模型相比，当批大小较大时，吞吐量更大。下表展示了有关数据： 请求每秒 (QPS) 量化 效率评估 速度 (tokens/s) 测试HuggingFace ``transformers`` 时的环境配置： 测试vLLM时的环境配置： 数据由vLLM吞吐量测试脚本测得，可通过以下命令复现 本部分介绍Qwen2模型（原始模型和量化模型）的效率测试结果，包括推理速度(tokens/s)与不同上下文长度时的显存占用(GB)。 Qwen2.5结果待更新，由于模型结构差异有限，Qwen2结果可供参考。 速度 (tokens/s) 我们测试生成2048 tokens时的速度与显存占用，输入长度分别为1、6144、14336、30720、63488、129024 tokens。(超过32K长度仅有 Qwen2-72B-Instuct 与 Qwen2-7B-Instuct 支持) batch size 设置为1，使用 GPU 数量尽可能少 [默认设定]=(gpu_memory_utilization=0.9 max_model_len=32768 enforce_eager=False) [设定 1]=(gpu_memory_utilization=0.98 max_model_len=4096 enforce_eager=True) [设定2] [设定 2]=(gpu_memory_utilization=1.0 max_model_len=4096 enforce_eager=True) [设定3] [设定 3]=(gpu_memory_utilization=1.0 max_model_len=8192 enforce_eager=True) 