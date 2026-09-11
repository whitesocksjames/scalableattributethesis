# Scalable Attribute Thesis

基于 Unicorn Part II 的 **full-resolution quality-scalable extension**：每个
formal operating point 提供 full-resolution **Base** 与 **Full**。最终机制为
`native truncated Unicorn prefix → learned BaseSynthesis → full-resolution Base
→ conditional Enhancement → Full`。这项贡献扩展了 Unicorn Part II 的 lossy
attribute progressive-refinement framework；不声称首次提出 layered coding，也不
声称 Original Unicorn 不支持 progressive decoding。

Formal static RGB experiment 已冻结：20 samples，Original `180/180`，Ours
`140/140`，总计 `320/320 FORMAL_REUSABLE` logical evaluations 和 460 decoded
endpoints。最终结果入口是
[formal_static_rgb_final_20260911](results/comparisons/formal_static_rgb_final_20260911/README.md)。

## 30 秒导航

| 我要找什么？ | 直接打开 |
| --- | --- |
| 当前模型 / 新增 Base module | [CURRENT_ARCHITECTURE](docs/repository/CURRENT_ARCHITECTURE.md) |
| Frozen checkpoints / 2K rescue / 4K joint | [CURRENT_OPERATING_POINTS](docs/repository/CURRENT_OPERATING_POINTS.md) · [machine-readable registry](configs/scalable_attribute/current_candidates.json) |
| 当前训练入口 | [CURRENT_TRAINING_ENTRYPOINTS](docs/repository/CURRENT_TRAINING_ENTRYPOINTS.md) |
| Formal evaluation contract / runtime 定义 | [CURRENT_EVALUATION_ENTRYPOINTS](docs/repository/CURRENT_EVALUATION_ENTRYPOINTS.md) |
| Final tables / RD 图 / evidence | [CURRENT_RESULT_INDEX](docs/repository/CURRENT_RESULT_INDEX.md) |
| 哪些 active / historical / broken？ | [Repository inventory](docs/repository/INVENTORY.md) |
| 本次整理范围 | [Phase 1 plan](docs/repository/PHASE1_PLAN.md) |
| N30 / HPC 使用规则 | [N30](N30_GUIDE.md) · [HPC](HPC_GUIDE.md) |

Frozen Ours curve 固定为 `512/1K/2K/4K/8K/16K/32K`。2K 使用 U-PATH
Base step500 + D111 Enhancement step1500；4K 使用 8K→4K joint step3000
Base/Full。256 仅保留为 historical screening evidence，不属于 final curve。

非当前的 dynamic/lossless/geometry 源码已收起到
[archive/upstream_out_of_scope](archive/upstream_out_of_scope/)。
原路径、依赖检查和恢复方法见 [archive](archive/README.md)。

旧 `canonical_operating_points.json` 是兼容历史 CLI 的 recipe，不能作为 formal
checkpoint authority。Formal identity 以 `formal_static_rgb_v1_checkpoints.json`
和 frozen registry 为准。Coding/rate/metric convention 与 Original Unicorn 对齐；
模型权重、数据和大型运行输出不作为源码资产。`drafts/` 和 dated result packages
中的历史证据仍保留；CURRENT 页面与 final package 是 examiner-facing authority。
下面的 Unicorn README 是明确标注的 upstream 原文归档，其中性能主张属于原作者，
不是本 thesis 对 Ours 的主张。

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
