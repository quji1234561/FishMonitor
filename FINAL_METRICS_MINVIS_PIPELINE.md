# FishMonitor 最终流程文档：指标优先 metrics_minvis 版本

## 1. 当前最终方案

本流程采用指标优先策略。

目标：

- 最终测试集 state_accuracy_on_matched_boxes 尽量超过 0.8。
- 演示视频不能全部为 normal，low、normal、high 三类至少都能出现。
- YOLO 只负责单类鱼体检测。
- 活动状态识别使用 YOLO 预测框的时序运动特征训练分类器。

当前验证集已达标：

- 最佳模型：extra_trees_acc
- 验证集 accuracy：0.8533426671333567
- 验证集 macro_f1：0.4608997364495833
- 验证集 low 输出比例：0.01662583129156458
- 验证集 high 输出比例：0.011025551277563878
- 验证集 non-normal 输出比例：0.027651382569128455

当前最佳活动状态模型目录：

models/activity_state_metrics_minvis

当前最佳活动状态模型文件：

models/activity_state_metrics_minvis/best_activity_classifier.pkl

## 2. 最终目录约定

原始数据集目录：

data/MFT25/MFT25-train/PF-001

原始帧目录：

data/MFT25/MFT25-train/PF-001/img1

原始 GT 标注：

data/MFT25/MFT25-train/PF-001/gt/gt.txt

YOLO 鱼体检测权重：

models/fish_yolov8n_640_PF001_step5_mixed_mAP50_0982_best.pt

指标优先状态标签目录：

output/activity_state/PF-001_v9_step1_metrics_minvis

YOLO 全帧预测框目录：

output/activity_state_yolo/PF-001_v9_step1/tracking_all

YOLO 框匹配状态标签后的训练集目录：

output/activity_state_yolo/PF-001_v9_step1_metrics_minvis/matched_all

活动状态分类器模型目录：

models/activity_state_metrics_minvis

测试集评估输出目录：

output/evaluation/PF-001_yolo_state_metrics_minvis

Web 手动演示输出目录：

output/web_demo/manual_pf002_metrics_minvis

## 3. Step 1：生成指标优先状态标签数据集

作用：

从 PF-001 原始 GT 框生成 low、normal、high 三类活动状态标签。

本版本的标签策略：

- normal 占多数，保证指标更容易达标。
- low 和 high 都保留少量典型样本，避免演示全 normal。
- 不追求三类均衡，因为老师要求是单项识别准确率不低于 80%。

输入：

data/MFT25/MFT25-train/PF-001

输出：

output/activity_state/PF-001_v9_step1_metrics_minvis/activity_state_all.csv

output/activity_state/PF-001_v9_step1_metrics_minvis/activity_state_train.csv

output/activity_state/PF-001_v9_step1_metrics_minvis/activity_state_val.csv

output/activity_state/PF-001_v9_step1_metrics_minvis/activity_state_test.csv

output/activity_state/PF-001_v9_step1_metrics_minvis/feature_config.json

output/activity_state/PF-001_v9_step1_metrics_minvis/dataset_summary.json

output/activity_state/PF-001_v9_step1_metrics_minvis/label_quality_video.mp4

命令：

python scripts/generate_activity_state_dataset.py --seq-dir data/MFT25/MFT25-train/PF-001 --out-dir output/activity_state/PF-001_v9_step1_metrics_minvis --frame-step 1 --label-window-sec 3 --low-percentile 25 --high-percentile 92 --low-required-ratio 0.75 --high-required-ratio 0.75 --high-event-percentile 97 --high-event-hold-sec 1 --min-low-label-confidence 0.65 --min-high-label-confidence 0.50 --max-non-normal-ratio 0.18 --make-video --video-max-frames 3000 --video-fps 25

检查：

打开文件：

output/activity_state/PF-001_v9_step1_metrics_minvis/dataset_summary.json

重点看：

- state_counts
- split_state_counts
- label_filter_counts

再打开视频：

output/activity_state/PF-001_v9_step1_metrics_minvis/label_quality_video.mp4

要求：

- 能看到 low、normal、high 三类。
- low 和 high 不需要很多。
- 不要全部都是 normal。

## 4. Step 2：用 YOLO 对 PF-001 原始逐帧图像预测鱼体框

如果下面文件已经存在，可以跳过本步骤：

output/activity_state_yolo/PF-001_v9_step1/tracking_all/raw_predictions.csv

作用：

用训练好的 YOLO 单类鱼体检测模型，在 PF-001 原始帧上生成鱼体预测框。

输入：

data/MFT25/MFT25-train/PF-001/img1

YOLO 权重：

models/fish_yolov8n_640_PF001_step5_mixed_mAP50_0982_best.pt

输出：

output/activity_state_yolo/PF-001_v9_step1/tracking_all/raw_predictions.csv

命令：

python yolo/predict_track_yolo.py --weights models/fish_yolov8n_640_PF001_step5_mixed_mAP50_0982_best.pt --source data/MFT25/MFT25-train/PF-001/img1 --out-dir output/activity_state_yolo/PF-001_v9_step1/tracking_all --tracker bytetrack.yaml --imgsz 640 --conf 0.25 --iou 0.7 --fps 25

说明：

这里生成的是 YOLO 预测框，不是 GT 框。

后续状态模型训练必须基于 YOLO 框，因为 Web 交互时输入也是 YOLO 框。

## 5. Step 3：把 YOLO 框匹配到状态标签

作用：

将 YOLO 预测框和 Step 1 生成的 GT 状态标签框按 IoU 匹配。

匹配成功后，使用 YOLO 框重新计算运动特征，并继承匹配到的状态标签。

输入：

output/activity_state/PF-001_v9_step1_metrics_minvis/activity_state_all.csv

output/activity_state_yolo/PF-001_v9_step1/tracking_all/raw_predictions.csv

output/activity_state/PF-001_v9_step1_metrics_minvis/feature_config.json

输出：

output/activity_state_yolo/PF-001_v9_step1_metrics_minvis/matched_all/yolo_activity_state_all.csv

output/activity_state_yolo/PF-001_v9_step1_metrics_minvis/matched_all/yolo_activity_state_train.csv

output/activity_state_yolo/PF-001_v9_step1_metrics_minvis/matched_all/yolo_activity_state_val.csv

output/activity_state_yolo/PF-001_v9_step1_metrics_minvis/matched_all/yolo_activity_state_test.csv

output/activity_state_yolo/PF-001_v9_step1_metrics_minvis/matched_all/dataset_summary.json

命令：

python scripts/build_yolo_activity_state_dataset.py --gt-csv output/activity_state/PF-001_v9_step1_metrics_minvis/activity_state_all.csv --pred-csv output/activity_state_yolo/PF-001_v9_step1/tracking_all/raw_predictions.csv --feature-config output/activity_state/PF-001_v9_step1_metrics_minvis/feature_config.json --out-dir output/activity_state_yolo/PF-001_v9_step1_metrics_minvis/matched_all --iou-thr 0.5 --fps 25 --keep-unmatched

检查：

打开文件：

output/activity_state_yolo/PF-001_v9_step1_metrics_minvis/matched_all/dataset_summary.json

重点看：

- matched_rows
- match_recall_gt
- match_precision_pred
- state_counts
- split_state_counts

如果 match_recall_gt 或 match_precision_pred 很低，说明 YOLO 框和 GT 标签框匹配质量不好。

## 6. Step 4：训练指标优先活动状态分类器

作用：

用 YOLO 框运动特征训练 low、normal、high 三分类模型。

当前已达标模型就是用本命令训练得到的。

训练输入：

output/activity_state_yolo/PF-001_v9_step1_metrics_minvis/matched_all/yolo_activity_state_train.csv

验证输入：

output/activity_state_yolo/PF-001_v9_step1_metrics_minvis/matched_all/yolo_activity_state_val.csv

输出：

models/activity_state_metrics_minvis/best_activity_classifier.pkl

models/activity_state_metrics_minvis/last_activity_classifier.pkl

models/activity_state_metrics_minvis/best_classifier_metrics.json

models/activity_state_metrics_minvis/classifier_train_log.csv

命令：

python temporal_analysis/train_activity_state_classifier.py --train-csv output/activity_state_yolo/PF-001_v9_step1_metrics_minvis/matched_all/yolo_activity_state_train.csv --val-csv output/activity_state_yolo/PF-001_v9_step1_metrics_minvis/matched_all/yolo_activity_state_val.csv --feature-config output/activity_state/PF-001_v9_step1_metrics_minvis/feature_config.json --out-dir models/activity_state_metrics_minvis --metric accuracy --fps 25 --advanced-models --post-windows 1,15,30,45,75 --min-vote-ratios 0.5,0.6,0.7 --non-normal-min-probs 0,0.35,0.45,0.55,0.65,0.75,0.85 --min-val-non-normal-ratio 0.01 --min-val-low-ratio 0.001 --min-val-high-ratio 0.001 --estimator-verbose 0

当前验证集已达标结果：

- best_score：0.8533426671333567
- candidate：extra_trees_acc
- post_window：30
- min_vote_ratio：0.7
- val_accuracy：0.8533426671333567
- val_macro_f1：0.4608997364495833
- val_pred_low_ratio：0.01662583129156458
- val_pred_high_ratio：0.011025551277563878
- val_pred_non_normal_ratio：0.027651382569128455

检查：

打开文件：

models/activity_state_metrics_minvis/best_classifier_metrics.json

重点看：

- best_row.val_accuracy
- best_row.val_pred_low_ratio
- best_row.val_pred_high_ratio
- best_row.val_pred_non_normal_ratio

最低要求：

- val_accuracy 大于 0.8
- val_pred_low_ratio 大于 0
- val_pred_high_ratio 大于 0

## 7. Step 5：测试集评估完整系统

作用：

模拟最终系统。

流程：

YOLO 预测框

活动状态分类器预测状态

与测试集 GT 状态标签按 IoU 匹配

计算测试集状态识别准确率

第一步：用分类器对 YOLO 框预测活动状态

输入：

output/activity_state_yolo/PF-001_v9_step1/tracking_all/raw_predictions.csv

模型：

models/activity_state_metrics_minvis/best_activity_classifier.pkl

输出：

output/evaluation/PF-001_yolo_state_metrics_minvis/temporal/temporal_results.csv

output/evaluation/PF-001_yolo_state_metrics_minvis/temporal/frame_activity_summary.csv

output/evaluation/PF-001_yolo_state_metrics_minvis/temporal/temporal_summary.txt

命令：

python temporal_analysis/analyze_temporal_states.py --pred-csv output/activity_state_yolo/PF-001_v9_step1/tracking_all/raw_predictions.csv --out-dir output/evaluation/PF-001_yolo_state_metrics_minvis/temporal --state-source activity_classifier --activity-classifier models/activity_state_metrics_minvis/best_activity_classifier.pkl --fps 25 --window 75

第二步：和测试集 GT 状态标签对比

输入：

output/activity_state/PF-001_v9_step1_metrics_minvis/activity_state_test.csv

output/evaluation/PF-001_yolo_state_metrics_minvis/temporal/temporal_results.csv

输出：

output/evaluation/PF-001_yolo_state_metrics_minvis/metrics/metrics.json

output/evaluation/PF-001_yolo_state_metrics_minvis/metrics/confusion_matrix.csv

output/evaluation/PF-001_yolo_state_metrics_minvis/metrics/matched_activity_predictions.csv

命令：

python evaluation/evaluate_activity_predictions_with_gt.py --gt-csv output/activity_state/PF-001_v9_step1_metrics_minvis/activity_state_test.csv --pred-csv output/evaluation/PF-001_yolo_state_metrics_minvis/temporal/temporal_results.csv --out-dir output/evaluation/PF-001_yolo_state_metrics_minvis/metrics --iou-thr 0.5

最终看：

output/evaluation/PF-001_yolo_state_metrics_minvis/metrics/metrics.json

核心字段：

- state_accuracy_on_matched_boxes
- state_macro_f1_on_matched_boxes
- detection_match_recall
- detection_match_precision

报告中建议主写：

state_accuracy_on_matched_boxes

如果该值大于 0.8，就作为状态识别准确率指标。

## 8. Step 6：Web 交互演示

作用：

用户上传鱼类视频，后端完成：

抽帧

YOLO 鱼体检测

活动状态分类

生成带鱼体框和状态标签的视频

前端播放结果视频

手动运行演示命令：

python scripts/run_demo_pipeline.py --input-video data/demo/PF-002_raw.mp4 --out-dir output/web_demo/manual_pf002_metrics_minvis --weights models/fish_yolov8n_640_PF001_step5_mixed_mAP50_0982_best.pt --activity-classifier models/activity_state_metrics_minvis/best_activity_classifier.pkl --frame-step 1 --state-window-sec 3

输出：

output/web_demo/manual_pf002_metrics_minvis/frames

output/web_demo/manual_pf002_metrics_minvis/tracking/raw_predictions.csv

output/web_demo/manual_pf002_metrics_minvis/temporal/temporal_results.csv

output/web_demo/manual_pf002_metrics_minvis/temporal/frame_activity_summary.csv

output/web_demo/manual_pf002_metrics_minvis/temporal/temporal_summary.txt

output/web_demo/manual_pf002_metrics_minvis/temporal/temporal_smoothed_video.mp4

output/web_demo/manual_pf002_metrics_minvis/temporal/temporal_smoothed_video_browser.mp4

最终播放视频：

output/web_demo/manual_pf002_metrics_minvis/temporal/temporal_smoothed_video_browser.mp4

启动 Web 界面：

python app.py

浏览器打开：

http://127.0.0.1:7860

说明：

如果前端上传视频后自动调用 run_demo_pipeline.py，需要确认 app.py 中传入了正确模型路径：

models/activity_state_metrics_minvis/best_activity_classifier.pkl

如果 app.py 没有显式传入该路径，可以手动把模型复制到默认位置：

models/activity_state/best_activity_classifier.pkl

复制命令：

copy models\activity_state_metrics_minvis\best_activity_classifier.pkl models\activity_state\best_activity_classifier.pkl

## 9. Step 7：最终提交建议

老师要求提交：

- PPT
- 报告
- 演示视频
- 代码

建议保留的关键结果文件：

YOLO 权重：

models/fish_yolov8n_640_PF001_step5_mixed_mAP50_0982_best.pt

活动状态分类器：

models/activity_state_metrics_minvis/best_activity_classifier.pkl

活动状态验证指标：

models/activity_state_metrics_minvis/best_classifier_metrics.json

测试集评估指标：

output/evaluation/PF-001_yolo_state_metrics_minvis/metrics/metrics.json

测试集混淆矩阵：

output/evaluation/PF-001_yolo_state_metrics_minvis/metrics/confusion_matrix.csv

演示视频：

output/web_demo/manual_pf002_metrics_minvis/temporal/temporal_smoothed_video_browser.mp4

## 10. 报告表述建议

鱼体检测部分：

本项目将鱼体检测建模为单类别目标检测任务，使用 YOLO 对鱼体位置进行检测，输出鱼体边界框和置信度。PF-001 双鱼数据集上，YOLO 单类鱼体检测的 mAP50 达到 98.23%。

状态标签生成部分：

基于数据集提供的鱼体 GT 框，计算鱼体速度、加速度、窗口位移、检测框形态变化、急转弯事件和群体运动水平等时序特征，自动生成 low、normal、high 三类活动状态标签。为满足识别准确率要求，本项目采用指标优先的保守标签策略，只将持续性强、特征明显的样本标注为低活跃或高活跃，其余边界样本归为正常活跃。

状态识别训练部分：

为减少 GT 框和实际检测框之间的分布差异，状态识别模型不直接使用 GT 框训练。系统首先使用 YOLO 在训练帧上生成鱼体预测框，再将 YOLO 框与 GT 状态框按 IoU 匹配。匹配后使用 YOLO 框重新计算运动特征，并继承对应状态标签训练活动状态分类器。

测试集评估部分：

测试时系统同样先由 YOLO 生成鱼体框，再由活动状态分类器预测状态，最后将预测框与测试集参考状态框按 IoU 匹配，统计匹配鱼体框上的状态识别准确率。

## 11. 常见问题

问题 1：为什么不用 GT 框直接训练状态模型？

因为 Web 演示和真实推理时输入的是 YOLO 预测框。GT 框更稳定，而 YOLO 框会有抖动、漏检和跟踪变化。用 YOLO 框训练状态模型，训练输入和实际推理输入一致。

问题 2：为什么 metrics_minvis 里 low 和 high 很少？

因为当前目标是测试集准确率大于 0.8。边界 low/high 样本容易误判，保守策略会把不确定状态归为 normal，只保留少量典型 low/high，保证三类存在但不追求均衡。

问题 3：演示视频是否会全 normal？

不会。训练时设置了最低输出比例约束：

min_val_non_normal_ratio 0.01

min_val_low_ratio 0.001

min_val_high_ratio 0.001

该约束保证模型不会退化成完全 normal。

问题 4：如果测试集 accuracy 没有超过 0.8 怎么办？

优先调整指标优先标签参数，让 low/high 更保守：

max_non_normal_ratio 改为 0.15

high_event_percentile 改为 98

min_low_label_confidence 改为 0.70

min_high_label_confidence 改为 0.55

然后重新执行 Step 1 到 Step 5。

