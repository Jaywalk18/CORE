# Copyright 2023 solo-learn development team.

from typing import Any, Dict, List, Optional, Sequence, Tuple

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.methods.base import BaseMethod
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class ObjCon(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """实现 ObjCon - 目标检测引导的对比学习框架
        
        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): 投影特征的维度
                proj_hidden_dim (int): 投影器隐藏层的神经元数量
                pred_hidden_dim (int): 预测器隐藏层的神经元数量
                detection_pretrained (bool): 是否使用预训练的检测模型
                det_model_type (str): 目标检测模型类型
                edge_detection_stage (bool): 是否启用边缘检测阶段
                semantic_stage (bool): 是否启用语义分割阶段
                lambda_global (float): 全局对比损失的权重
                lambda_local (float): 局部对比损失的权重
        """
        
        super().__init__(cfg)
        
        # 从配置中提取参数
        self.proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        self.proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        self.pred_hidden_dim: int = cfg.method_kwargs.pred_hidden_dim
        
        # 可选参数
        self.detection_pretrained = cfg.method_kwargs.get("detection_pretrained", True)
        self.det_model_type = cfg.method_kwargs.get("det_model_type", "faster_rcnn")
        self.edge_detection_stage = cfg.method_kwargs.get("edge_detection_stage", True)
        self.semantic_stage = cfg.method_kwargs.get("semantic_stage", False)
        self.lambda_global = cfg.method_kwargs.get("lambda_global", 1.0)
        self.lambda_local = cfg.method_kwargs.get("lambda_local", 1.0)
        
        # 投影器
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, self.proj_hidden_dim),
            nn.BatchNorm1d(self.proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.proj_hidden_dim, self.proj_output_dim),
        )
        
        # 预测器
        self.predictor = nn.Sequential(
            nn.Linear(self.proj_output_dim, self.pred_hidden_dim),
            nn.BatchNorm1d(self.pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.pred_hidden_dim, self.proj_output_dim),
        )
        
        # 边缘检测模块
        if self.edge_detection_stage:
            self.edge_detector = self._create_edge_detector()
        
        # 目标检测/分割模块
        if self.semantic_stage:
            self.det_model = self._create_detection_model()
            
        # ROI特征提取器
        self.roi_feature_extractor = nn.Sequential(
            nn.Linear(self.features_dim, 512),
            nn.ReLU()
        )
    
    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """添加方法特定的默认值/检查配置

        Args:
            cfg (omegaconf.DictConfig): DictConfig对象

        Returns:
            omegaconf.DictConfig: 与参数相同，用于避免错误
        """

        cfg = super(ObjCon, ObjCon).add_and_assert_specific_cfg(cfg)

        # 断言必需的参数存在
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.pred_hidden_dim")

        # 设置可选参数的默认值
        if omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.detection_pretrained"):
            cfg.method_kwargs.detection_pretrained = True
            
        if omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.det_model_type"):
            cfg.method_kwargs.det_model_type = "faster_rcnn"
            
        if omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.edge_detection_stage"):
            cfg.method_kwargs.edge_detection_stage = True
            
        if omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.semantic_stage"):
            cfg.method_kwargs.semantic_stage = False
            
        if omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.lambda_global"):
            cfg.method_kwargs.lambda_global = 1.0
            
        if omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.lambda_local"):
            cfg.method_kwargs.lambda_local = 1.0

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """将投影器和预测器参数添加到父类的可学习参数中

        Returns:
            List[dict]: 可学习参数列表
        """
        
        extra_learnable_params = [
            {"name": "projector", "params": self.projector.parameters()},
            {"name": "predictor", "params": self.predictor.parameters()},
            {"name": "roi_feature_extractor", "params": self.roi_feature_extractor.parameters()},
        ]
        
        if self.edge_detection_stage:
            extra_learnable_params.append(
                {"name": "edge_detector", "params": self.edge_detector.parameters()}
            )
            
        # 注意：目标检测模型通常设置为固定参数，不进行微调
        
        return super().learnable_params + extra_learnable_params

    def _create_edge_detector(self) -> nn.Module:
        """创建边缘检测模块
        
        Returns:
            nn.Module: 边缘检测模型
        """
        # 简单的边缘检测模块实现
        edge_detector = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1)
        )
        return edge_detector

    def _create_detection_model(self) -> nn.Module:
        """创建目标检测模型
        
        Returns:
            nn.Module: 目标检测模型
        """
        if self.det_model_type == "faster_rcnn":
            model = fasterrcnn_resnet50_fpn(pretrained=self.detection_pretrained)
            # 冻结模型参数
            for param in model.parameters():
                param.requires_grad = False
            return model
        else:
            raise ValueError(f"不支持的检测模型类型: {self.det_model_type}")
    
    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        """执行骨干网络和投影器的前向传递

        Args:
            X (torch.Tensor): 图像批次

        Returns:
            Dict[str, Any]: 包含父类输出和投影特征的字典
        """
        
        out = super().forward(X)
        z = self.projector(out["feats"])
        p = self.predictor(z)
        
        out.update({"z": z, "p": p})
        
        # 如果启用边缘检测
        if self.edge_detection_stage and not self.training:
            edges = self.edge_detector(X)
            out.update({"edges": edges})
            
        # 如果启用语义分割/目标检测
        if self.semantic_stage and not self.training:
            with torch.no_grad():
                det_out = self.det_model(X)
            out.update({"det_out": det_out})
        
        return out
    
    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """ObjCon 的训练步骤

        Args:
            batch (Sequence[Any]): 数据批次，格式为 [img_indexes, [X], Y]，其中 [X] 是包含图像批次的列表
            batch_idx (int): 批次索引

        Returns:
            torch.Tensor: 总损失
        """
        
        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        
        # 获取视图 1 和视图 2 (经过数据增强的同一张图像)
        X = batch[1]
        x1, x2 = X[0], X[1]
        
        # 前向传播
        feats1 = self.backbone(x1)
        feats2 = self.backbone(x2)
        
        # 全局特征投影和预测
        z1 = self.projector(feats1)
        z2 = self.projector(feats2)
        
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        
        # 全局对比损失 (类似 BYOL)
        global_loss = (
            F.cosine_similarity(p1, z2.detach(), dim=-1).mean() +
            F.cosine_similarity(p2, z1.detach(), dim=-1).mean()
        ) / 2
        global_loss = -global_loss * self.lambda_global
        
        # 局部对比损失
        local_loss = 0.0
        if self.edge_detection_stage or self.semantic_stage:
            # 这里是局部对比损失的具体实现
            # 可以基于边缘检测或目标检测结果提取局部特征
            if self.edge_detection_stage:
                edges1 = self.edge_detector(x1)
                edges2 = self.edge_detector(x2)
                # 使用边缘信息引导局部特征提取和对比
                # ...
                
            if self.semantic_stage:
                # 使用目标检测模型提取区域
                with torch.no_grad():
                    det_out1 = self.det_model(x1)
                    det_out2 = self.det_model(x2)
                # 提取局部 ROI 特征并计算对比损失
                # ...
                
            local_loss = local_loss * self.lambda_local
        
        # 记录各个损失组件
        self.log("train_global_loss", global_loss, on_epoch=True, sync_dist=True)
        self.log("train_local_loss", local_loss, on_epoch=True, sync_dist=True)
        
        # 总损失
        total_loss = class_loss + global_loss + local_loss
        
        return total_loss
