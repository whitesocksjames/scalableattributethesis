# Scalable Attribute Thesis

基于 Unicorn Part II 的两层质量可伸缩 RGB Attribute compression：每个 operating
point 提供 full-resolution **Base** 与 **Full**。当前是阶段性开发整理，不是 final freeze。

## 30 秒导航

| 我要找什么？ | 直接打开 |
| --- | --- |
| 当前模型 / 新增 Base module | [CURRENT_ARCHITECTURE](docs/repository/CURRENT_ARCHITECTURE.md) |
| 当前 checkpoints / 2K rescue / 4K joint | [CURRENT_OPERATING_POINTS](docs/repository/CURRENT_OPERATING_POINTS.md) · [machine-readable registry](configs/scalable_attribute/current_candidates.json) |
| 当前训练入口 | [CURRENT_TRAINING_ENTRYPOINTS](docs/repository/CURRENT_TRAINING_ENTRYPOINTS.md) |
| 当前 hard / external evaluation | [CURRENT_EVALUATION_ENTRYPOINTS](docs/repository/CURRENT_EVALUATION_ENTRYPOINTS.md) |
| 当前汇总数据和 RD 图 | [CURRENT_RESULT_INDEX](docs/repository/CURRENT_RESULT_INDEX.md) |
| 哪些 active / historical / broken？ | [Repository inventory](docs/repository/INVENTORY.md) |
| 本次整理范围 | [Phase 1 plan](docs/repository/PHASE1_PLAN.md) |
| N30 / HPC 使用规则 | [N30](N30_GUIDE.md) · [HPC](HPC_GUIDE.md) |

当前 2K：U-PATH Base step500 + D111 Enhancement step1500。
当前 4K：8K→4K joint step3000 Base/Full；Base bitrate 不要求低于 8K。
256 保留 evaluation。所有点仍是 working candidates，不因历史报告写有 FREEZE 而自动封版。

旧 `canonical_operating_points.json` 是兼容历史 CLI 的 official-mapping recipe，
不能直接作为 current candidate registry。尤其不要用旧 `--point 2k/4k` 选择新权重。
Coding/rate/metric convention 与 Original Unicorn 对齐；模型权重、数据和大型运行输出
不作为源码资产。`drafts/`/`results/` 中的历史证据仍保留，其路径可能被分析脚本引用。

<details>
<summary>Original upstream Unicorn README (retained attribution and historical setup)</summary>

# Unicorn: A Versatile Point Cloud Compressor Using Universal Multiscale Conditional Coding

## Abstract

A universal multiscale conditional coding framework, Unicorn, is proposed to compress the geometry and attribute of any given point cloud. Geometry compression is addressed in [Part I](https://ieeexplore.ieee.org/document/10682571) of this paper, while attribute compression is discussed in [Part II](https://ieeexplore.ieee.org/document/10682566).

For geometry compression, we construct the multiscale sparse tensors of each voxelized point cloud frame and properly leverage lower-scale priors in the current and (previously processed) temporal reference frames to improve the conditional probability approximation or content-aware predictive reconstruction of geometry occupancy in compression.

For attribute compression, Since attribute components exhibit very different intrinsic characteristics from the geometry element, e.g., 8-bit RGB color versus 1-bit occupancy, we process the attribute residual between lower-scale reconstruction and current-scale data. Similarly, we leverage spatially lower-scale priors in the current frame and (previously processed) temporal reference frame to improve the probability estimation of attribute intensity through conditional residual prediction in lossless mode or enhance the attribute reconstruction through progressive residual refinement in lossy mode for better performance.

The proposed Unicorn is a versatile, learning-based solution capable of compressing static and dynamic point clouds with diverse source characteristics in both lossy and lossless modes. Following the same evaluation criteria, Unicorn significantly outperforms standard-compliant approaches like MPEG G-PCC, V-PCC, and other learning-based solutions, yielding state-of-the-art compression efficiency while presenting affordable complexity for practical implementations.

For more information, please visit our homepage: https://njuvision.github.io/Unicorn/ 

## Environment

* pytorch, MinkowskiEngine, etc. 
    * You can use docker to simply configure the environment: `docker pull jianqiang1995/pytorch:1.10.0-cuda11.1-cudnn8-devel`


## Dataset

* **ShapeNet**: https://shapenet.org/ 
* **RWTT**: https://texturedmesh.isti.cnr.it/ 
* **MPEG Dataset (Static Objects)**: http://mpegfs.int-evry.fr/MPEG/PCC/DataSets/pointCloud/CfP/datasets/ (MPEG password is required) 
(You can also access some of them on our NJU BOX. ( https://box.nju.edu.cn/d/51327ae7c2644c0fa1c4/ ))
* **MPEG Dataset (Dynamic Objects)**: https://mpeg-pcc.org/index.php/pcc-content-database/
* **KITTI**: https://www.cvlibs.net/datasets/kitti/
* **Ford**: https://mpegfs.int-evry.fr/ws-mpegcontent/MPEG-I/Part05-PointCloudCompression/dataSets_new/Dynamic_Acquisition/Ford  (MPEG password is required) 
(You can also access some of them on our NJU BOX. ( https://box.nju.edu.cn/d/2739fe997265478c8673/ ))


(Note: The training dataset generation methods and the amount of training dataset are not required to be fixed. We provide some examples in `data_utils/datasets/README.sh` to show how to perform sampling, partition, quantization, and other operations on raw mesh or point cloud data to generate the training datasets.)

## Pretrained Models

* **Geometry** **ckpt**: [https://box.nju.edu.cn/f/57095bcf44604dc2baed/?dl=1](https://box.nju.edu.cn/d/5393fa52f42c4d918576/)
* **Attribute** **ckpt**: [https://box.nju.edu.cn/f/57095bcf44604dc2baed/?dl=1](https://box.nju.edu.cn/f/7162590c2a46489291bd/)

## Results

`./results`


## Authors

These files are provided by Nanjing University [Vision Lab](https://vision.nju.edu.cn/). Thanks to Prof. Dandan Ding from Hangzhou Normal University and Prof. Yi Lin from Fudan University for their help. Please contact us (mazhan@nju.edu.cn) if you have any questions.

</details>
