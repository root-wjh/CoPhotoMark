# 扩展WIDER：登记人脸数与严格集合可用率

已扫描579文件，按内容及像素去重后578源图，按源图划分20%验证、80%测试。410图成功登记，168登记失败保留记录；测试中323图成功登记，其中223图所有检测人脸均被登记。三条件clean/JPEG Q95/JPEG Q50、auto/blind两模式，共2460条核验记录，无运行异常。64位载荷仍为8位人数+8位编号+48位tag；阈值预先固定auto=1、blind=0.8125，不随N或测试表现改变。未运行错误密钥评估，因此本次仅回答可用率，不重新证明FAR。

推荐正文采用table_face_count_full_coverage.tex：所有检测人脸都登记的子集，避免部分登记与额外不合格人脸混淆。全部成功登记的323图结果必须保留，见table_face_count.tex及natural_by_face_count.csv；登记失败见enrollment_coverage.csv。完整覆盖是源图登记覆盖标准，不按核验是否接受筛选。不得隐藏总体结果或把223称为所有测试图。

完整覆盖子集原图盲接受率：1人73.6%(72图)、2人71.4%(105图)、3人70.6%(17图)、4-6人66.7%(27图)、7+人50.0%(2图)。观察趋势支持可用率代价，但最高人数样本仍少，且自然照片人数与质量相关，不能当作因果证明。

额外质量对照：完整覆盖测试图按最小登记人脸短边64-127/128-255/256+、最小拉普拉斯方差<50/50-199/200+联合分层，四个人数分组1/2/3/4+每层取相同图数，按source哈希选图不看分数，得到每组10图共40图。结果并非全部单调，因此不能只展示下降的曲线。此控制是粗分层而非彻底消除质量差异，见quality_matched_by_face_count.csv和matched_sources.csv。

Q50在当前严格整图判决下接受率为0，必须保留。其与旧BAR指标不是同一指标；不能将本次协议视为全部旧实验的数值复现。

另报告any_expected_face_evidence_pct：检测结果中至少一张脸具有正确登记人数、范围内编号和达阈值tag。它是机制诊断，不代表对应编号与真实身份绑定，也不是合法替代认证规则。完整覆盖原图中其随人数保持较高，而严格集合成功降低，可用于解释多脸“全部恢复”的代价。

论文Section 4.7在两模式比较后插入完整覆盖表格，并加：

```latex
\textcolor{blue}{We stratify the expanded WIDER test set by
enrolled face count and report both all successful enrollments
and the subset with complete detected-face coverage.
In the latter, clean-image blind acceptance decreases from
73.6\% for one face to 66.7\% for four to six faces,
consistent with the requirement to recover every enrolled
number. Thresholds are fixed across groups; the highest-count
group contains only two images.}
```

中文：按登记人数对扩展测试集分组，同时报告全部成功登记图和覆盖全部检测人脸的子集。后者原图盲接受率从单脸73.6%降至4-6脸66.7%，与必须恢复全部编号的要求一致。各组阈值固定，最高人数分组仅2图。

Response: Thank you for this comment. We have expanded the WIDER evaluation and report acceptance rates stratified by enrolled face count under fixed thresholds. We include all successfully enrolled test images and separately report complete detected-face coverage, together with enrollment failures and group sample sizes. In the complete-coverage subset, clean-image blind acceptance decreases from 73.6% for one face to 66.7% for four to six faces, supporting the availability cost of strict set verification. A supplementary quality-stratified comparison is also provided.

运行：python scripts/run_wider_expanded.py --output <新的输出目录>。默认扫描D:/CoPhotoMark_Revision/assets/wider，逐图检查点可断点重启；新增图片后必须另建目录（输入哈希变更会拒绝混用）。源图与原代码不改动。本次运行实际发生在D:/code-skills/test-ppt/wider_expanded/run_20261114，交付副本在runs/wider_expanded_20261114，因此JSON路径仍指实际执行位置，这是来源记录，不应伪造执行路径。
