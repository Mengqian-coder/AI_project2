# 实验二：搜索算法

本项目包含四道必做题和两个拓展实验，使用 Python 实现 A*、Minimax 与 Alpha-beta，
并比较启发函数、搜索深度和动作访问顺序对结果与搜索开销的影响。

仓库提交六个实验程序、对应测试、复现实验的数据与图表，以及两份文档：
本 README 和 [AI 实验二报告](AI实验二报告.docx)。
README 说明环境、程序入口、复现方式和数据组织；报告统一呈现实验设计、结果与思考。

## 1. 环境与依赖

- Python：代码使用 Python 3.10 及以上版本的语法，报告中的原始计时数据采集于 Python 3.14.7。
- 四道必做题及全部正确性测试：仅依赖 Python 标准库，测试框架为 `unittest`。
- 拓展实验绘图：需要 Matplotlib，本次验证版本为 3.11.2。

安装绘图依赖：

```powershell
python -m pip install -r requirements.txt
```

克隆仓库后，以下命令均在仓库根目录执行。
无需配置外部数据路径；
默认样例在源码中定义，拓展实验按固定种子生成输入。
使用 `python 程序名.py --help` 可查看对应参数。

## 2. 本地程序与实验对应关系

| 实验 | 程序 | 内容 |
| --- | --- | --- |
| 题目一 | `q1_maze.py` | 单位移动代价网格的 A* 最短路径 |
| 题目二 | `q2_weighted.py` | 带权网格的 A* 最小移动代价 |
| 题目三 | `q3_alpha_beta.py` | Minimax、Alpha-beta 及访问顺序比较 |
| 题目四 | `q4_treasure.py` | A* 距离预计算与双人宝藏博弈 |
| 拓展一 | `q2_heuristic_experiment.py` | 探究曼哈顿启发函数与有限步前瞻比较 |
| 拓展二 | `q4_depth_experiment.py` | 探究浅搜索何时选错首步、损失多少 |

四道必做题分别独立实现；拓展一导入题目二，拓展二导入题目四。

```text
AI_project2/
├── README.md
├── AI实验二报告.docx            # 正式综合实验报告
├── requirements.txt            # 绘图依赖
├── .gitignore
├── q1_maze.py
├── q2_weighted.py
├── q3_alpha_beta.py
├── q4_treasure.py
├── q2_heuristic_experiment.py
├── q4_depth_experiment.py
├── test_*.py                    # 六个对应的正确性测试脚本
└── output/
    ├── q2_heuristics/           # 拓展一的地图、数据与图表
    └── q4_depth/                # 拓展二的案例、数据与图表
```

程序、测试脚本和 `output/` 数据共同用于复现；关键数据和图表纳入综合实验报告。
虚拟环境、临时文件、缓存和工作草稿不纳入仓库。

## 3. 四道必做题的运行与结果核对

### 3.1 题目一：A* 迷宫最短路径

```powershell
python q1_maze.py
python q1_maze.py --trace
python q1_maze.py --compare
```

默认输出官方样例的一条最短路径：

```text
[(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)]
```

包含起终点，共 7 次移动。`--trace` 显示有效出队顺序及 `g/h/f`；
`--compare` 比较 `h=0` 与曼哈顿距离。官方样例中，两者路径均为 7 步，
扩展节点数分别为 15、13。

接口：`astar(start, goal, grid)` 返回坐标路径，无解返回 `[]`。
坐标采用 `(row, col)`，四方向移动，每步代价为 1，`X` 为障碍。

### 3.2 题目二：带权 A* 搜索

```powershell
python q2_weighted.py
python q2_weighted.py --trace
python q2_weighted.py --compare
```

默认输出官方样例的最小总代价 `7`。`--trace` 还显示路径、每步进入费用及累计费用；
`--compare` 比较 `h=0` 与修正曼哈顿启发函数。官方样例中，两者代价均为 7，
扩展节点数分别为 14、10。

接口：`astar_weighted(start, goal, grid)` 返回最小总代价，无解返回 `-1`。
移动收取进入格子的费用，初始起点不收费，进入终点免费。
默认启发函数为 `h=m*max(Manhattan-1, 0)`，其中 `m` 是非终点可通行格的最小进入费用。

### 3.3 题目三：Minimax 与 Alpha-beta

```powershell
python q3_alpha_beta.py
python q3_alpha_beta.py --trace
python q3_alpha_beta.py --compare
python q3_alpha_beta.py --compare --order-experiment
```

默认输出官方样例的最佳得分 `5`。`--trace` 显示递归及边界更新；
`--compare` 显示两种算法的访问与剪枝统计；
`--order-experiment` 增加仅反转 MIN 节点孩子顺序的比较。

默认按从左到右顺序搜索。官方样例中，Minimax 与 Alpha-beta 的访问次数分别为 15、14，
叶子评估次数分别为 8、7。调整访问顺序后，Alpha-beta 访问 12 次、评估 6 次叶子，根得分仍为 5。

接口：`minimax(root)`、`alpha_beta(root)` 返回根节点分值，
树通过 `TreeNode(value=..., children=...)` 构造。

### 3.4 题目四：双人宝藏博弈

```powershell
python q4_treasure.py
python q4_treasure.py --trace
python q4_treasure.py --compare
python q4_treasure.py --first-moves
python q4_treasure.py --compare --order-experiment
```

默认输出 A、B 与四个宝藏之间的最小移动代价矩阵，以及两种算法在 H=2、H=4 时的结果。

| 搜索深度 | A 最优首步 | 该深度下的最优收益差 U |
| --- | --- | ---: |
| H=2 | T2 | -1 |
| H=4 | T3 | 3 |

H 表示双方合计的行动数：H=2 为 A→B，H=4 为 A→B→A→B。
每次收益为“宝藏价值−移动代价”，评价值为 `U=score_a-score_b`；
达到搜索深度时直接使用当前 U。负收益动作仍然合法，规则中没有跳过回合动作。

`--trace` 展示一条最优应对序列的移动、收益和累计分数；
`--compare` 展示 Minimax 与 Alpha-beta 的开销；
`--first-moves` 精确计算每个首步的分值；
`--order-experiment` 比较默认、反向和即时收益排序。参数可以组合使用。

主要接口为 `compute_distances()`、`GameState`、
`minimax(state, distances, depth)` 和 `alpha_beta(state, distances, depth)`。
博弈搜索返回 `SearchResult(best_target, best_value, line)`。

**题目二与题目四的收费规则不同：**题目四的 A、B、宝藏格进入费用均为 1，
目标宝藏格也收费，因此使用 `h=m*Manhattan`。
经过其他宝藏不自动收集，只有被选中的目标宝藏从剩余集合中移除。

上述四个程序将结果输出到终端，不自动保存到 `output/`。

## 4. 两个拓展实验的复现

### 4.1 拓展一：带权搜索的启发函数比较

完整实验：

```powershell
python q2_heuristic_experiment.py
```

默认比较 H0 与一步、两步、三步前瞻，覆盖 54 张地图，包含 16、32、64 三种边长，
每个方法每张图重复计时 7 轮，基础随机种子为 `20261028`。
每次查询使用新缓存，将地图准备、启发计算和 A* 搜索计入总时间。

这里的 H0、H1、H2、H3 表示**启发函数的前瞻层数**；
题目四的 H=2、H=4 表示**博弈树搜索深度**。

快速试跑，并将结果写入单独目录：

```powershell
python q2_heuristic_experiment.py --sizes 8 16 --seeds-per-family 1 --repeats 3 --output tmp/q2_pilot
```

该脚本默认生成图表，需要 Matplotlib。完整结果保存在
`output/q2_heuristics/`：

| 文件 | 内容 |
| --- | --- |
| `maps.json` | 完整输入地图及种子 |
| `results.json` | 参数、运行环境、逐轮计时、搜索统计与汇总 |
| `comparison.png` | 估值质量、扩展数和耗时对照图 |

已保存的实验中，H0 的总耗时最低；前瞻使估值更紧，但节约的搜索不足以抵消额外计算。
耗时受机器与运行负载影响，重新计时的数值可能变化，分析时应结合中位数与四分位距。

### 4.2 拓展二：浅搜索何时选错首步

完整实验：

```powershell
python q4_depth_experiment.py
```

固定题目四的地图与移动规则，生成 1481 条分组案例记录：
官方设置 1 条、四条单变量扫描共 80 条、T2/T3 二维组合 400 条、随机组 1000 条。
扫描价值范围为 1–20，随机组种子为 `20261103`。

对每个首步分别计算 Q2、Q4，以完整四次行动的目标衡量浅搜索首步：

```text
首步损失 regret = max_a Q4(a) - Q4(a_H2)
```

Q4(a) 表示固定首步为 a 后，双方完整最优应对所得的最终收益差。
损失大于 0 才表示次优；首步不同但最终同样最优，损失仍为 0。
同时区分浅层最优动作“全部正确”“并列敏感”“全部次优”，避免结果受平局规则混淆。
原题中固定浅搜索首步 T2 后完整应对得 1，完整最优值为 3，因此损失为 2。

**与报告 4.5 节对应的平局策略比较：**脚本保持 H=2 的评分不变，
仅在 A 的根节点最优首步同分时应用下列规则：

| 策略 | 同分时如何选择 |
| --- | --- |
| `default` | 按 T1、T2、T3、T4 选第一个 |
| `reverse` | 按上述编号选最后一个 |
| `nearest` | 选从 A 出发最小移动代价的宝藏 |
| `gain` | 选“宝藏价值−从 A 出发移动代价”最大的宝藏 |

`nearest`、`gain` 再次同分时按编号选第一个。这些规则只在 H=2 最优集合中选择，
不改变递归访问顺序，也不使用 Q4 选择动作；Q4 仅用于事后评价损失。
注意：此处的根平局规则与题目四为 Alpha-beta 剪枝设置的全树动作排序是两项不同的比较。

固定 T1=9、T4=7，遍历 T2、T3 各为 1～20 的 400 组组合，结果为：

| 根平局策略 | 次优首步数 | 平均首步损失 |
| --- | ---: | ---: |
| `default` | 27 | 0.290 |
| `reverse` | 21 | 0.190 |
| `nearest` | 20 | 0.170 |
| `gain` | 23 | 0.230 |

以上完整实验命令会自动生成四种策略的结果，无需额外参数。

只生成文本与数据，不绘图：

```powershell
python q4_depth_experiment.py --no-plots --output tmp/q4_depth_no_plots
```

默认输出到 `output/q4_depth/`：

| 文件 | 内容 |
| --- | --- |
| `cases.json` | 参数、地图、距离矩阵和全部案例输入 |
| `results.json` | 全部首步分值、最优动作集合、应对序列与损失；每案例含 `tie_policies` |
| `results.csv` | 便于筛选的逐案例数据 |
| `summary.json` | 各输入组的汇总统计及四种平局策略结果 |
| `tie_policies.csv` | 每案例每策略一行，含选择、完整价值、损失及相对默认规则的变化 |
| `value_scans.png` | 四条单变量扫描的首步损失 |
| `decision_map.png` | T2/T3 二维组合的损失与并列分类 |

各组输入设计不同，且可能重复包含相同价值向量，应分别统计。
结论针对当前地图和当前 U 截断评价；此实验衡量一次首步选择的损失，
后续双方使用完整最优应对。

两个拓展脚本的 `--output` 均可更改输出位置；
重新运行会更新目标目录中的同名结果文件，试跑时可使用上面的独立目录。

## 5. 正确性测试

在项目根目录运行全部测试：

```powershell
python -m unittest discover -v
```

当前共有 **72 项测试**，成功结束时输出 `Ran 72 tests` 和 `OK`。
测试按固定随机种子运行，使用独立参考算法进行核对。

| 测试脚本 | 测试数 | 主要验证内容 |
| --- | ---: | --- |
| `test_q1_maze.py` | 8 | 与 BFS 比较最短距离，检查路径合法性及代价更新 |
| `test_q2_weighted.py` | 10 | 与 Dijkstra 比较，检查免费终点、绕路、零权与无解 |
| `test_q3_alpha_beta.py` | 12 | 独立树求值，检查边界、剪枝、共享节点及访问统计 |
| `test_q4_treasure.py` | 17 | 距离与博弈参考求解，检查状态转移、深度及动作顺序 |
| `test_q2_heuristic_experiment.py` | 8 | 可采纳性、一致性、前瞻估值、缓存及最优代价 |
| `test_q4_depth_experiment.py` | 17 | 独立排列枚举、首步损失、并列分类、序列回放及四种根平局规则 |

需要定位某个实验时，指定对应测试模块即可，例如：

```powershell
python -m unittest -v test_q4_depth_experiment
```

测试验证实现正确性；性能数据和探索结论通过第 3、4 节的实验命令复现。
测试运行本身不会重新生成拓展实验的图表或计时数据。

## 6. 统计口径

- A* 的 `expanded` 统计实际枚举邻居的节点，排除终点及跳过的旧条目；
  `pushed` 包含起点与重复入堆条目。
- 博弈搜索的 `visited` 包含根和叶子，按递归访问次数计；
  `expanded` 为实际生成动作的非叶子，`leaves` 为终止评估次数，
  满足 `visited=expanded+leaves`。
- `cutoffs` 只统计确实跳过至少一个剩余孩子的剪枝；
  `skipped_children` 只统计跳过的直接孩子数。
- 题目四的博弈统计不包含 A* 距离预计算及额外首步分值分析。
  拓展一以完整查询耗时的中位数与四分位距比较性能，计入地图准备、启发计算和搜索，
  每次查询使用新缓存。

## 7. 仓库提交内容

| 文档 | 内容 |
| --- | --- |
| README | 环境与依赖、六个实验的运行入口、数据位置及统一测试方法 |
| [AI 实验二报告](AI实验二报告.docx) | 覆盖六个实验的目标、原理、实现、实验设置、结果与图表、对照分析、拓展探索和总结 |

文档之外，提交六个实验源码、六个对应测试脚本、`requirements.txt`、`.gitignore`，
以及第 4 节列出的输入、结果与图表。拓展一保留报告采用的原始计时数据，
重新运行得到的耗时应视为本机新测量，不要求与报告小数逐位相同。
六个实验的分析统一组织在综合实验报告中，保证报告可以独立阅读。
