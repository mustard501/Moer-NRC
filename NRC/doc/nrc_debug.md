多视角支持格式
NRCCollectMain 支持三种 views_file 格式（内容都表示多个 camera.transform）：
```
A. 直接数组
[
  { "position":[...], "look_at":[...], "up":[...] },
  { "position":[...], "look_at":[...], "up":[...] }
]
```
```
B. views 包装
{
  "views": [
    { "transform": { "position":[...], "look_at":[...], "up":[...] } },
    { "transform": { "position":[...], "look_at":[...], "up":[...] } }
  ]
}
```
```
C. camera_transforms 包装
{
  "camera_transforms": [
    { "position":[...], "look_at":[...], "up":[...] }
  ]
}
```
scene.json 里新增的 NRC 配置（读取于 renderer.nrc）
示例（可加在场景 renderer 下）：
```
"nrc": {
  "dataset_dir": "NRC/runtime/datasets",
  "dataset_prefix": "classroom",
  "views_file": "nrc_views.json",
  "threads": 12,
  "flush_threshold": 32768,
  "max_bounce_record": 8,
  "global_keep": 1.0,
  "bounce_keep_probs": [1.0, 0.5, 0.25, 0.25, 0.1, 0.1, 0.1, 0.1]
}
```
说明：

views_file 路径相对 scene_dir 解析
若不填 views_file，默认只采当前 scene.json 的单个相机
写文件时按 bounce 做 keep/drop：
- `bounce_keep_probs[i]` 对应 bounce=`i+1` 的保留概率
- `bounce > max_bounce_record` 直接丢弃
- 最终保留概率 = `bounce_keep_probs * global_keep`（再 clamp 到 [0,1]）
- 不写这些字段时使用上述默认值
运行方式
构建后执行：
```
Moer-NRC-Collect <scene_dir>
```
例如：

```
Moer-NRC-Collect scenes/classroom
```
输出数据默认到：
```
NRC/runtime/datasets/<dataset_prefix>_view_0.bin
NRC/runtime/datasets/<dataset_prefix>_view_1.bin
...
NRC/runtime/<scene_rel>/<dataset_prefix>_bounce_stats.txt
```

`*_bounce_stats.txt` 记录每个视角各 bounce 的 **written / candidates**（写入时实时累计，每完成一个视角刷新一次），用于检查 keep/drop 后的分布，无需遍历 `.bin`。