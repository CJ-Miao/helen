# Helen 空pileup修复 - 写入assembly序列

## 问题背景

Helen polish结果中大量序列包含超长T-run。根本原因：
- 重复序列/低复杂度区域的reads天然MAPQ很低
- marginpolish配置`filterAlignmentsWithMapQBelowThisThreshold: 60`过滤了所有reads
- 生成全零pileup图像
- Helen模型面对全零图像时预测出 T碱基×RLE=10（训练数据中无"无数据"模式）

## 数据格式

- **MarginPolish图像**: HDF5 (.h5) 格式，每个image shape = (SEQ_LENGTH, IMAGE_HEIGHT)，uint8
- **Helen预测**: HDF5 (.hdf) 格式，存储在predictions/contig/chunk_name_prefix/chunk_id/{position,bases,rles}
- **Dataloader返回**: `(contig, contig_start, contig_end, chunk_id, image, position, is_empty_pileup, hdf5_filepath)`

## 修复方案

在预测阶段遇到空pileup时，直接写入原始assembly序列到prediction HDF5文件，而不是让模型产生伪预测。

### 参数传递链

```
PolishInterface.polish_genome(assembly_fasta)
  → CallConsensusInterface.call_consensus(assembly_fasta)
    → predict_cpu.predict(assembly_fasta)  / predict_gpu.predict(assembly_fasta)
```

### 修改的文件

| 文件 | 修改内容 |
|------|---------|
| `helen/modules/python/PolishInterface.py` | call_consensus调用传入assembly_fasta；perform_stitch调用移除assembly参数 |
| `helen/modules/python/CallConsensusInterface.py` | call_consensus签名+参数传递 |
| `helen/modules/python/models/predict_cpu.py` | assembly FASTA加载 + 空pileup写assembly逻辑 |
| `helen/modules/python/models/predict_gpu.py` | 与CPU版本相同的逻辑 |
| `helen/modules/python/Stitch.py` | 删除small_chunk_stitch的assembly_seq参数、>80%单一碱基fallback逻辑、create_consensus_sequence的assembly_dict参数 |
| `helen/modules/python/StitchInterface.py` | 删除perform_stitch的assembly_fasta参数和assembly_dict加载逻辑 |
| `helen/helen.py` | 删除stitch子命令的--assembly参数定义和perform_stitch调用参数 |

### 核心逻辑

**空pileup检测** (已在dataloader_predict.py中实现):
```python
MIN_EFFECTIVE_ROWS = 50
effective_rows = np.sum(np.any(image != 0, axis=1))
is_empty_pileup = effective_rows < MIN_EFFECTIVE_ROWS
```

**空pileup处理** (predict_cpu.py / predict_gpu.py中修改):
```python
# 编码assembly碱基为label格式
base_labels = np.zeros(len(pos_i), dtype=npuint8)
rle_labels = np.ones(len(pos_i), dtype=np.uint8)
for j in range(len(pos_i)):
    p = pos_i[j][0]
    if 0 <= p < len(assembly_seq):
        b = assembly_seq[p]
        base_labels[j] = {'A': 1, 'C': 2, 'G': 3, 'T': 4}.get(b, 0)
prediction_data_file.write_prediction(contig, contig_start, contig_end, chunk_id,
                                       pos_i, base_labels, rle_labels, filename)
```

碱基映射: A→1, C→2, G→3, T→4, 未知→0

两处写入点:
1. **Batch级别**: 整个batch都是空pileup时，写入所有sample的assembly序列
2. **Per-image级别**: 单个image是空pileup时，写入该image的assembly序列

### 脚本位置

```
/home/miaocj/docker_dir/consolida/helen/helen/modules/python/models/predict_cpu.py
/home/miaocj/docker_dir/consolida/helen/helen/modules/python/models/predict_gpu.py
/home/miaocj/docker_dir/consolida/helen/helen/modules/python/CallConsensusInterface.py
/home/miaocj/docker_dir/consolida/helen/helen/modules/python/PolishInterface.py
```

## 注意事项

1. 需要传入`assembly_fasta`参数给polish命令，指向原始assembly文件
2. 支持`.gz`压缩的FASTA文件
3. assembly字典按contig名称查找，需要注意h5文件中contig命名与FASTA header的对应关系
4. 碱基值0表示未知碱基（非A/C/G/T），对应RLE=1
5. `import numpy as np` 已添加到predict_cpu.py和predict_gpu.py（修复缺失导入）

## 待做

- [ ] 用现有数据验证修复效果，确认空pileup区域不再出现T-run

---

## 2026-05-09 追加：全空pileup contig过滤

### 问题

之前修改后，空pileup图像写入了assembly序列。如果一个contig的所有images都是空pileup（即没有normal model prediction），这个contig仍然会在prediction HDF5中有数据，stitching阶段会产生错误的consensus（全是assembly序列，没有polish）。

用户要求：当一个contig全部都是空pileup images时，完全过滤掉这个contig。

### 修改方案

预测阶段维护两个追踪集合：
- `all_contigs_seen`: 遇到的所有contig（空pileup和normal prediction都追踪）
- `contigs_with_predictions`: 有至少一个normal model prediction的contig

函数末尾，计算`all_empty_contigs = all_contigs_seen - contigs_with_predictions`，从HDF5文件中删除这些contig的所有数据（包括predictions组和meta记录），stitching阶段自动跳过这些contig。

### 三层处理策略

| 情况 | 处理阶段 | 处理方式 | 覆盖对象 |
|------|---------|---------|---------|
| **全空contig** | 预测阶段 | 从HDF5删除，不出现在consensus中 | 所有images都是空pileup的contig |
| **部分空contig** | 预测阶段 | 空pileup images写assembly序列 | 部分images是空pileup的contig |
| **信号差非空contig** | Stitching阶段 | 单一碱基>80%回退assembly | 模型预测出全T但images非全零的contig |

三层互不重叠，覆盖所有场景。

### 核心逻辑

### 修改的文件

| 文件 | 修改内容 |
|------|---------|
| `predict_cpu.py` | 添加contig追踪 + 全空contig删除逻辑 |
| `predict_gpu.py` | 与CPU版本相同 |

### 核心逻辑

```python
# 追踪集合初始化（在with torch.no_grad()之前）
contigs_with_predictions = set()
all_contigs_seen = set()

# 全空batch处理 - 更新all_contigs_seen
if batch_empty:
    if assembly_dict is not None:
        for i in range(len(contig)):
            contig_name = ...
            all_contigs_seen.add(contig_name)  # 追踪
            ...

# per-image空pileup处理 - 更新all_contigs_seen
if is_empty_pileup[i]:
    contig_name = ...
    all_contigs_seen.add(contig_name)  # 追踪
    ...

# normal prediction - 更新contigs_with_predictions
contig_name = ...
contigs_with_predictions.add(contig_name)
prediction_data_file.write_prediction(...)

# 函数末尾 - 删除全空contig数据
all_empty_contigs = [c for c in all_contigs_seen if c not in contigs_with_predictions]
if all_empty_contigs:
    for contig_name in all_empty_contigs:
        if contig_name in prediction_data_file.file_handler.get('predictions', {}):
            del prediction_data_file.file_handler['predictions'][contig_name]
    if 'predictions' in prediction_data_file.file_handler:
        for contig_name in all_empty_contigs:
            prediction_data_file.meta['predictions'] -= {...}
            prediction_data_file.meta['predictions_contig'] -= {...}
```

---

## 2026-05-11 追加：弱pileup检测 + 滑动窗口assembly回退

### 问题

tmp2结果中仍有8个T-run（6个为模型假阳性），虽然contig整体T%不高（25-50%），但局部区域有超长T-run（20-750bp）：

| Contig | T-run长度 | 全局T% | 模型预测可靠性 |
|--------|----------|--------|--------------|
| contig_103 | 750bp | 25-40% | 假阳性 |
| contig_30 | 700bp | 25-34% | 假阳性 |
| contig_31 | 600bp | 53% | 假阳性 |
| contig_131 | 425bp | 25-37% | 假阳性 |
| contig_49 | 351bp | 51% | 假阳性 |
| contig_62 | 304bp | 35% | 假阳性 |

**根本原因**：这些contig的pileup存在（有效行数>=50，不被标记为空），但信号非常稀疏：
- contig_103: 一个chunk只有113/1000有效行，密度0.001
- contig_49: 每行只有1个非零像素，密度~0.011
- 正常pileup密度: ~0.022-0.029

模型对弱pileup（非全零但信号不足）产生全T预测，因为：
1. 模型在正常pileup上训练，没见过弱pileup
2. 弱pileup是OOD输入，模型权重碰巧输出T高概率
3. 不是代码写了"默认写T"，是模型自身行为

### 修改方案

在预测阶段增加**整图密度检测**：非零密度 < 0.5% 的样本标记为weak，滑动窗口中该样本的模型输出被assembly序列替换。

### 核心逻辑

```python
WINDOW_QUALITY_THRESHOLD = 0.005  # 非零密度阈值

# 1. 每batch检测每个sample的密度
images_np = images.cpu().numpy()
sample_is_weak = torch.zeros(batch_size, dtype=torch.bool)
for si in range(batch_size):
    w_density = np.sum(images_np[si] != 0) / images_np[si].size
    if w_density < WINDOW_QUALITY_THRESHOLD:
        sample_is_weak[si] = True

# 2. 预计算weak sample的assembly碱基labels
assembly_base_labels_for_weak = ...  # shape: (batch, seq_len)

# 3. 滑动窗口中，模型推理后，替换weak sample的输出
for window in sliding_windows:
    output_base, output_rle, hidden = model(image_chunk, hidden)
    if sample_is_weak[si]:
        # 用assembly碱基的one-hot替换模型输出
        output_base[si, :, :] = assembly_one_hot
        output_rle[si, :] = 1.0

# 4. 经过softmax + 滑动窗口聚合，assembly碱基最终胜出
```

### 修改的文件

| 文件 | 修改内容 |
|------|---------|
| `predict_cpu.py` | 整图密度检测 + pre-compute assembly + 滑动窗口替换 |
| `predict_gpu.py` | 与CPU版本相同 |

### 密度阈值说明

- 正常pileup: 密度 ~2.2-2.9%
- 弱pileup (contig_49): 密度 ~1.1%
- 极弱pileup (contig_103): 密度 ~0.1%
- 阈值 0.5%: 区分正常和弱pileup的中间值

### 预期效果

弱pileup样本不再产生T×RLE=10假阳性，consensus中contig局部T-run应消失。
