数据集路径`/mnt/s3files/s3-us-west2-default/dataprocessing/raw/opendata/Sekai-game` 具体看memory proposed.md

Total samples: 1618

=== location (9 unique values) ===
  Le Prarion, Chamonix-Mont-Blanc, France               238  (14.71%)
  Castle Rock Beach, Meelup, Australia                  226  (13.97%)
  Lago di Braies, Braies, South Tyrol, Italy            184  (11.37%)
  Fushimi Inari Taisha, Kyoto, Japan                    170  (10.51%)
  East Maddon Park, London, United Kingdom              168  (10.38%)
  Mardi Himal Trail, Kaski, Nepal                       166  (10.26%)
  Shengshan Island, Zhoushan, China                     161  ( 9.95%)
  Mýrdalssandur, Southern Region, Iceland               158  ( 9.77%)
  Yamadera, Yamagata Prefecture, Japan                  147  ( 9.09%)

=== scene (3 unique values) ===
  outdoor-natural                                      1481  (91.53%)
  outdoor-urban                                         134  ( 8.28%)
  indoor-ancient                                          3  ( 0.19%)

=== crowdDensity (1 unique values) ===
  empty                                                1618  (100.00%)

=== weather (5 unique values) ===
  cloudy/foggy                                          895  (55.32%)
  sunny                                                 346  (21.38%)
  snowy                                                 279  (17.24%)
  rainy                                                  95  ( 5.87%)
  unknown                                                 3  ( 0.19%)

=== timeOfDay (5 unique values) ===
  day                                                   957  (59.15%)
  night                                                 487  (30.10%)
  sunset                                                170  (10.51%)
  unknown                                                 3  ( 0.19%)
  sunrise                                                 1  ( 0.06%)

====================================================================================================
Conditional Distribution: weather | location
====================================================================================================
location                                     | cloudy/foggy |        rainy |        snowy |        sunny |      unknown |    Total
----------------------------------------------------------------------------------------------------------------------------------
Castle Rock Beach, Meelup, Australia         |          123 |            0 |            0 |          103 |            0 |      226
East Maddon Park, London, United Kingdom     |           13 |           78 |            0 |           74 |            3 |      168
Fushimi Inari Taisha, Kyoto, Japan           |           67 |           16 |            0 |           87 |            0 |      170
Lago di Braies, Braies, South Tyrol, Italy   |           56 |            0 |          115 |           13 |            0 |      184
Le Prarion, Chamonix-Mont-Blanc, France      |          185 |            0 |            2 |           51 |            0 |      238
Mardi Himal Trail, Kaski, Nepal              |           11 |            0 |          142 |           13 |            0 |      166
Mýrdalssandur, Southern Region, Iceland      |          152 |            0 |            6 |            0 |            0 |      158
Shengshan Island, Zhoushan, China            |          159 |            1 |            0 |            1 |            0 |      161
Yamadera, Yamagata Prefecture, Japan         |          129 |            0 |           14 |            4 |            0 |      147

====================================================================================================
Conditional Distribution: timeOfDay | location
====================================================================================================
location                                     |          day |        night |      sunrise |       sunset |      unknown |    Total
----------------------------------------------------------------------------------------------------------------------------------
Castle Rock Beach, Meelup, Australia         |          103 |           56 |            0 |           67 |            0 |      226
East Maddon Park, London, United Kingdom     |           85 |           80 |            0 |            0 |            3 |      168
Fushimi Inari Taisha, Kyoto, Japan           |           88 |           82 |            0 |            0 |            0 |      170
Lago di Braies, Braies, South Tyrol, Italy   |          184 |            0 |            0 |            0 |            0 |      184
Le Prarion, Chamonix-Mont-Blanc, France      |          163 |           65 |            1 |            9 |            0 |      238
Mardi Himal Trail, Kaski, Nepal              |           56 |          106 |            0 |            4 |            0 |      166
Mýrdalssandur, Southern Region, Iceland      |           62 |           14 |            0 |           82 |            0 |      158
Shengshan Island, Zhoushan, China            |           77 |           76 |            0 |            8 |            0 |      161
Yamadera, Yamagata Prefecture, Japan         |          139 |            8 |            0 |            0 |            0 |      147

MOVE（受监督的平移动作）

┌──────────────────────────────┬──────────────────┬───────────┬────────────────┐
│             类别             │       占比       │ 覆盖视频  │    平均段长    │
├──────────────────────────────┼──────────────────┼───────────┼────────────────┤
│ go forward                   │ 80.3%            │ 1602/1618 │ 156 帧（~5秒） │
├──────────────────────────────┼──────────────────┼───────────┼────────────────┤
│ no-op（站立）                │ 9.0%             │ 1156      │ 66 帧          │
├──────────────────────────────┼──────────────────┼───────────┼────────────────┤
│ forward+left / forward+right │ 3.5% / 4.7%      │ ~1180     │ ~16 帧         │
├──────────────────────────────┼──────────────────┼───────────┼────────────────┤
│ 纯左/右侧移                  │ 各 ~1.0%         │ ~640      │ ~19 帧         │
├──────────────────────────────┼──────────────────┼───────────┼────────────────┤
│ 后退类合计                   │ ~0.7%            │ ~270      │ 短段           │
├──────────────────────────────┼──────────────────┼───────────┼────────────────┤
│ uncertain                    │ 0.004%（112 帧） │ 24        │ —              │
└──────────────────────────────┴──────────────────┴───────────┴────────────────┘

VIEW（视角动作）


┌────────────────────────┬───────────────────────────┬──────────┐
│          类别          │           占比            │ 覆盖视频 │
├────────────────────────┼───────────────────────────┼──────────┤
│ no-op                  │ 72.4%                     │ 全部     │
├────────────────────────┼───────────────────────────┼──────────┤
│ turn left / turn right │ 11.9% / 11.9%（完全对称） │ 1593     │
├────────────────────────┼───────────────────────────┼──────────┤
│ 上下看 + 四个对角      │ 各 ~0.6-0.8%，合计 ~3.9%  │ ~1100    │
└────────────────────────┴───────────────────────────┴──────────┘

联合：前进+不转头 60.7%；前进+左/右转 16.5%；站立不动 5.9%。

视频间差异（每视频 forward 占比）：中位 84.9%，但 p10 只有 56%、最小 0%——全量里存在不少大量站立/环顾的视频，比前 20 个样本（同一场景连续切片）多样得多。

对训练的启示

1. 动作严重不均衡：forward 与 view no-op 占绝对主导。模型会很擅长"往前走"，但后退（0.7%）、侧移（2%）、抬头低头（<2%）的可控性只能靠少量样本学；如果这些动作对你重要，后续可考虑按动作稀有度加权采样 chunk。
2. 左右完全对称（turn left 11.85% vs right 11.90%），无方向偏置，好事。
3. uncertain 几乎不出现（112 帧）——当前阈值下标签很"确定"，如果想让模型学到论文里的 uncertainty-aware 行为，需要调 ActionLabelConfig 放宽 uncertain 判定。