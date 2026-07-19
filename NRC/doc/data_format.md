# NRC Dataset Binary Format (v1)

本文档描述 `Moer-NRC-Collect` 输出的 `.bin` 训练数据格式。实现见 `NRC/include/NRC/NRCTypes.h` 与 `NRC/src/NRCDatasetWriter.cpp`。

## 文件布局

```
┌──────────────────────────────────────┐
│ DatasetHeaderV1  (固定 96 字节)       │
├──────────────────────────────────────┤
│ RadianceSampleRecordV1  #0 (52 字节) │
│ RadianceSampleRecordV1  #1           │
│ ...                                  │
│ RadianceSampleRecordV1  #N-1         │
└──────────────────────────────────────┘
```

- 所有多字节整数与 `float` 均为 **小端序（little-endian）**。
- `float` 为 **IEEE 754 单精度**。
- 文件总大小应满足：`file_size == headerBytes + recordCount * 52`。

## 魔数与版本

| 字段   | 值        | 说明                         |
|--------|-----------|------------------------------|
| magic  | `NRC1`    | 4 字节 ASCII，即 `0x4E524331` |
| version| `1`       | 当前仅支持 v1                |

读取时应先校验 `magic` 与 `version`，再使用 `headerBytes` 定位记录区起始偏移。

---

## DatasetHeaderV1

逻辑字段如下（C++ 结构体在 MSVC x64 下因 `uint64_t` 对齐，`sizeof` 为 **96 字节**）：

| 偏移 | 类型        | 字段名         | 说明 |
|------|-------------|----------------|------|
| 0    | `char[4]`   | magic          | 固定 `"NRC1"` |
| 4    | `uint32`    | version        | 数据集版本，当前为 `1` |
| 8    | `uint32`    | headerBytes    | 头部字节数，当前为 `96` |
| 12   | `uint8[4]`  | *(padding)*    | 对齐填充，无意义 |
| 16   | `uint64`    | recordCount    | 样本记录条数 |
| 24   | `uint32`    | featureDim     | 特征维度，当前为 `9`（pos 3 + wo 3 + normal 3） |
| 28   | `uint32`    | labelDim       | 标签维度，当前为 `3`（RGB radiance） |
| 32   | `uint32`    | flags          | 位标志，见下表 |
| 36   | `uint32`    | reserved0      | 保留，写为 `0` |
| 40   | `float[3]`  | sceneAabbMin   | 场景 AABB 最小角（世界坐标） |
| 52   | `float[3]`  | sceneAabbMax   | 场景 AABB 最大角（世界坐标） |
| 64   | `uint32[8]` | reserved       | 保留，写为 `0` |

### flags 位定义

| 位 | 名称                 | 值   | 说明 |
|----|----------------------|------|------|
| 0  | `DatasetFlagHasNormal` | `1` | 记录中包含 `normal` 字段 |

v1 采集器始终写入法线，`flags` 通常为 `1`。

### Python `struct` 格式（头部）

```python
HEADER_FMT = "<4sII4xQIIII3f3f8I"  # size = 96
```

`struct.unpack` 将每个 `3f` 展开为 3 个独立 `float`，完整字段顺序为：

`(magic, version, header_bytes, record_count, feature_dim, label_dim, flags, reserved0, aabb_min_x, aabb_min_y, aabb_min_z, aabb_max_x, aabb_max_y, aabb_max_z, r0..r7)`

读取时需手动将 `aabb_min_*` / `aabb_max_*` 组合为 3 元组。

---

## RadianceSampleRecordV1

每条样本描述路径上的一个表面顶点及其沿出射方向的 GT radiance。

| 偏移 | 类型       | 字段名       | 说明 |
|------|------------|--------------|------|
| 0    | `float[3]` | position     | 世界坐标系下的交点位置 `(x, y, z)` |
| 12   | `float[3]` | outgoingDir  | 归一化出射方向 `wo`，指向路径上一顶点（bounce=1 时指向相机） |
| 24   | `float[3]` | normal       | 几何法线（`geometryNormal`） |
| 36   | `float[3]` | targetRGB    | 该顶点沿 `wo` 方向的 outgoing radiance RGB（线性 HDR，非 tonemap） |
| 48   | `uint16`   | bounce       | 路径深度，从 `1` 起计（首个非 null 表面交点为 1） |
| 50   | `uint16`   | _padding     | 对齐填充，写为 `0` |

单条记录固定 **52 字节**。

### bounce 含义

- `bounce = 1`：相机光线首个有效表面交点；`wo` 为指向相机的方向。
- `bounce = k`：路径上第 `k` 个有效表面顶点。
- `null` BSDF 表面会被跳过，不单独记样本，也不递增 bounce 计数后再写回（见 `NRCCollectorIntegrator::Li`）。

### targetRGB 含义

- 由路径上 **反向累加**（backward accumulation）得到的局部 outgoing radiance 估计。
- 仅当 `localRadiance` 非黑且无 NaN 时写入（见 `appendSample`）。
- 值为 Monte Carlo 估计，存在噪声；与渲染像素 RGB 在 bounce=1 时对应同一物理量，但单次样本会有方差。

### Python `struct` 格式（记录）

```python
RECORD_FMT = "<3f3f3f3fHH"  # size = 52
```

对应解包为 14 个标量（4 组 `float` + `bounce` + `padding`），需手动组合为向量。

---

## 文件命名约定

采集程序默认输出：

```
<dataset_dir>/<scene_rel_path>/<dataset_prefix>_view_<i>.bin
<dataset_dir>/<scene_rel_path>/<dataset_prefix>_bounce_stats.txt
```

例如：

```
NRC/runtime/scenes/classroom/scene_view_0.bin
NRC/runtime/scenes/classroom/scene_view_1.bin
NRC/runtime/scenes/classroom/scene_bounce_stats.txt
```

每个 `.bin` 对应一个相机视角（`camera.transform`）。

`*_bounce_stats.txt` 在写入样本时同步累计各 `bounce` 的 **written/candidates**（含 keep/drop 前后对比），每个视角采集结束后刷新；用于检查分布是否合理，无需再遍历巨型 `.bin`。格式示例：

```
# NRC bounce sample counts (after write-time keep/drop)
# written = kept samples, candidates = before keep/drop
# max_bounce_record=8 global_keep=1
# bounce_keep_probs=1,0.5,0.25,0.25,0.1,0.1,0.1,0.1

[view 0] file=scene_view_0.bin written=45000000 candidates=180000000
  bounce 1: written=35000000 candidates=35000000
  bounce 2: written=8000000 candidates=16000000
  ...

[summary] views=1 written_total=45000000 candidates_total=180000000
```

写时过滤由 `renderer.nrc` 控制：`max_bounce_record`、`bounce_keep_probs`、`global_keep`。

---

## 读取伪代码

```python
import struct

HEADER_FMT = "<4sII4xQIIII3f3f8I"
RECORD_FMT = "<3f3f3f3fHH"
HEADER_SIZE = 96
RECORD_SIZE = 52

with open(path, "rb") as f:
    header = struct.unpack(HEADER_FMT, f.read(HEADER_SIZE))
    magic, version, header_bytes, record_count = header[0], header[1], header[2], header[3]
    assert magic == b"NRC1" and version == 1

    for _ in range(record_count):
        rec = unpack_record(f.read(RECORD_SIZE))
        # rec = (position, outgoing_dir, normal, target_rgb, bounce, padding)
```

写入端在 `close()` 时会回写头部中的 `recordCount`；正常结束的文件应满足大小一致性校验。

---

## 相关工具

使用 `NRC/utils/nrc_inspect.py` 可在命令行检查 `.bin` 文件的头部信息与样本统计：

```bash
python NRC/utils/nrc_inspect.py path/to/scene_view_0.bin
python NRC/utils/nrc_inspect.py path/to/scene_view_0.bin --header-only
python NRC/utils/nrc_inspect.py path/to/scene_view_0.bin --rgb-stats --samples 3
```

大文件（数十 GB）建议先用 `--header-only` 查看元数据；完整 bounce 统计需扫描全部记录。
