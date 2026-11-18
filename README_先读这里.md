> 2026-11-05更新：用户真实clean日志确认检测人数变化与盲同步范围不足。先读 docs/clean失败诊断与修正_20261105.md，使用--clean-only开发对照；不要直接跑主实验。full64.json保持不变，新增phase_dev/step10_dev仅为待验证配置。

# CoPhotoMark 独立返修版（协议配置 v2，2026-11-03）

原始工程 C:/Users/dell'/PycharmProjects/CoPhotoMark 保持只读。此次实现修正不代表已经满足全部审稿意见；WIDER工程检查显示覆盖率、blind重同步与保水印边界修改仍有限制。

请按以下顺序阅读：

1. [严肃核查报告](docs/严肃核查报告_20261103.md)：论文与代码的实际不一致、已修复事项、真实试跑结果、文稿必须替换的位置及未完成要求。
2. [当前运行命令](docs/当前运行命令_20261103.md)：Windows清理/CPU、智星云GPU环境、WIDER20图、全子集、配对消融、单图与全部输出说明。
3. [完整逐条修改意见](docs/完整修改意见_逐条对应审稿人.md)：每条审稿意见与修改思路；已按当前实现更新。
4. [协议与边界](docs/协议与安全边界.md)。

## 当前默认协议

- QIM使用论文式(3) Q_a={2Δk+aΔ}，主Δ=18；配置version=2，旧码本/旧图/旧阈值不可混用，所有主表必须重跑。
- 64位是人数8+逻辑编号8+认证tag48；128位是8+8+112。结构字段精确检查，阈值只作用于tag。
- PNG内Y-LSB存储完整元数据与HMAC；默认不生成.cpm.json，也不从外部sidecar救援默认流程。--debug-metadata才额外保存副本。
- 图内metadata含claim副本，但验证者仍需要外部可信expected claim和私钥；PNG不是公开自包含核验凭证。
- 检测当前所有脸、验证完整唯一编号集；保留三态，UNDECIDABLE不能全部算作成功检出篡改。
- 删除/复制/插入/重排/替换/受控漏检等已实现；真实自然漏检和真实平台/截图未完成。attack/scenario_coverage.csv记录实际是否生成，skip不会隐藏。
- GPU仅用于检测，加载ONNX session后检查是否真的有CUDA；DCT、攻击及统计仍在CPU。

## 先自检

```powershell
cd 'D:\CoPhotoMark_Revision'
python -m pytest -q -p no:cacheprovider
python scripts/check_environment.py --provider cpu
```

完整运行命令见上面的专用文档。最新测试39项通过；本机WIDER首20张仅3张全覆盖多人登记，其中clean auto 2/3、blind 1/3接受，不能宣称大规模可靠性已解决。

## 输出和清理

每次run使用新空目录。旧runs/verification删除命令被桌面自动审批拒绝，用户可自行执行 scripts/clean_old_results.ps1；脚本仅删除明确列出的旧结果。当前源码ZIP不含旧结果、.venv、任何私钥或数据集。audit/20261103保留本次小规模核查记录，不作为论文主结果。
