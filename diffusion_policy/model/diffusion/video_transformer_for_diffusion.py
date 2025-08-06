from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)


class VideoTransformerForDiffusion(ModuleAttrMixin):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int = None,
        cond_dim_video_encoded: int = 2048,  # 编码后的图像特征维度
        cond_dim_traj: int = 9,  # 单个时间步的轨迹维度
        cond_time_steps: int = 16,  # T'，条件数据的时间步数
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = False,
        time_as_cond: bool = True,
        n_cond_layers: int = 0,
        # 新增参数用于控制位置编码策略
        preserve_traj_temporal: bool = True,  # 是否保留轨迹的时间结构
    ) -> None:
        super().__init__()

        # 计算条件维度
        self.cond_dim_video_encoded = cond_dim_video_encoded
        self.cond_dim_traj = cond_dim_traj
        self.cond_time_steps = cond_time_steps
        self.preserve_traj_temporal = preserve_traj_temporal

        # 总的条件维度
        total_cond_dim = cond_dim_video_encoded + cond_time_steps * cond_dim_traj

        # 计算token数量
        if n_obs_steps is None:
            n_obs_steps = horizon

        T = horizon
        T_cond = 1  # 时间token
        if not time_as_cond:
            T += 1
            T_cond -= 1

        # 条件处理策略
        self.has_cond = total_cond_dim > 0
        if self.has_cond:
            assert time_as_cond
            if preserve_traj_temporal:
                # 保留轨迹时间结构：1(time) + 1(image) + T'(traj_sequence)
                T_cond += 1 + cond_time_steps  # image token + trajectory tokens
            else:
                # 不保留时间结构：1(time) + 1(image) + 1(flattened_traj)
                T_cond += 2  # image token + flattened trajectory token

        # 输入嵌入
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # 条件嵌入
        self.time_emb = SinusoidalPosEmb(n_emb)

        # 图像和轨迹嵌入
        self.cond_image_emb = None
        self.cond_traj_emb = None

        if self.has_cond:
            if cond_dim_video_encoded > 0:
                self.cond_image_emb = nn.Linear(cond_dim_video_encoded, n_emb)

            if cond_dim_traj > 0:
                if preserve_traj_temporal:
                    # 单个时间步的轨迹嵌入
                    self.cond_traj_emb = nn.Linear(cond_dim_traj, n_emb)
                else:
                    # 展平的轨迹嵌入
                    self.cond_traj_emb = nn.Linear(cond_time_steps * cond_dim_traj, n_emb)

        # 条件位置编码
        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False

        if T_cond > 0:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))

            if preserve_traj_temporal and cond_time_steps > 0:
                # 为轨迹序列创建专门的时间位置编码
                self.traj_temporal_emb = nn.Parameter(torch.zeros(1, cond_time_steps, n_emb))
            else:
                self.traj_temporal_emb = None

            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4 * n_emb,
                    dropout=p_drop_attn,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_cond_layers)
            else:
                self.encoder = nn.Sequential(nn.Linear(n_emb, 4 * n_emb), nn.Mish(), nn.Linear(4 * n_emb, n_emb))

            # 解码器
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_layer)
        else:
            # 仅编码器模式
            encoder_only = True
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layer)

        # 注意力掩码
        if causal_attn:
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)

            if time_as_cond and self.has_cond:
                S = T_cond
                t, s = torch.meshgrid(torch.arange(T), torch.arange(S), indexing="ij")
                mask = t >= (s - 1)
                mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
                self.register_buffer("memory_mask", mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # 输出头
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        # 常量
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.encoder_only = encoder_only

        # 初始化
        self.apply(self._init_weights)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            SinusoidalPosEmb,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = ["in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight"]
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ["in_proj_bias", "bias_k", "bias_v"]
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, VideoTransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if hasattr(module, "cond_pos_emb") and module.cond_pos_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
            if hasattr(module, "traj_temporal_emb") and module.traj_temporal_emb is not None:
                torch.nn.init.normal_(module.traj_temporal_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def get_optim_groups(self, weight_decay: float = 1e-3):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # 位置编码参数不进行权重衰减
        no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        if hasattr(self, "cond_pos_emb") and self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")
        if hasattr(self, "traj_temporal_emb") and self.traj_temporal_emb is not None:
            no_decay.add("traj_temporal_emb")

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(
        self, learning_rate: float = 1e-4, weight_decay: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.95)
    ):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        video_cond: Optional[torch.Tensor] = None,
        traj_cond: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Args:
            sample: (B, T, input_dim) - 主要的动作序列
            timestep: (B,) or int - 扩散时间步
            video_cond: (B, cond_dim_image_encoded) - 重新格式化的条件数据
            traj_cond: (B, T', cond_dim_traj) - 轨迹条件数据（如果有）

        Returns:
            output: (B, T, output_dim)
        """
        batch_size = sample.shape[0]

        # 1. 时间嵌入
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(batch_size)
        time_emb = self.time_emb(timesteps).unsqueeze(1)  # (B, 1, n_emb)

        # 2. 处理输入序列
        input_emb = self.input_emb(sample)  # (B, T, n_emb)

        if self.encoder_only:
            # BERT模式：直接拼接时间和输入
            token_embeddings = torch.cat([time_emb, input_emb], dim=1)
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]
            x = self.drop(token_embeddings + position_embeddings)
            x = self.encoder(src=x, mask=self.mask)
            x = x[:, 1:, :]  # 移除时间token
        else:
            # 编码器-解码器模式

            # 3. 处理条件数据
            cond_embeddings = [time_emb]  # 开始于时间嵌入

            if self.has_cond and video_cond is not None:
                # 分离图像和轨迹数据
                if self.cond_dim_video_encoded > 0:
                    video_emb = self.cond_image_emb(video_cond).unsqueeze(1)  # (B, 1, n_emb)
                    cond_embeddings.append(video_emb)
                    # image_data = video_cond[:, : self.cond_dim_video_encoded]  # (B, cond_dim_image_encoded)
                    # image_emb = self.cond_image_emb(image_data).unsqueeze(1)  # (B, 1, n_emb)
                    # cond_embeddings.append(image_emb)

                if self.cond_dim_traj > 0:
                    # traj_data = video_cond[:, self.cond_dim_video_encoded :]  # (B, T' * cond_dim_traj)
                    traj_data = traj_cond

                    if self.preserve_traj_temporal:
                        # traj_data = traj_data.view(batch_size, self.cond_time_steps, self.cond_dim_traj)
                        traj_emb = self.cond_traj_emb(traj_data)  # (B, T', n_emb)

                        # 添加轨迹专用的时间位置编码
                        if self.traj_temporal_emb is not None:
                            traj_emb = traj_emb + self.traj_temporal_emb[:, : self.cond_time_steps, :]

                        cond_embeddings.append(traj_emb)
                    else:
                        # 不保留时间结构：作为单个token处理
                        traj_data = traj_data.view(batch_size, -1)  # (B, T' * cond_dim_traj)
                        traj_emb = self.cond_traj_emb(traj_data).unsqueeze(1)  # (B, 1, n_emb)
                        cond_embeddings.append(traj_emb)

            # 4. 合并条件嵌入
            cond_embeddings = torch.cat(cond_embeddings, dim=1)  # (B, T_cond, n_emb)

            # 5. 添加条件位置编码
            tc = cond_embeddings.shape[1]
            if self.cond_pos_emb is not None:
                position_embeddings = self.cond_pos_emb[:, :tc, :]
                cond_embeddings = cond_embeddings + position_embeddings

            # 6. 条件编码器
            cond_embeddings = self.drop(cond_embeddings)
            memory = self.encoder(cond_embeddings)  # (B, T_cond, n_emb)

            # 7. 主序列解码器
            t = input_emb.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]
            input_emb = input_emb + position_embeddings
            input_emb = self.drop(input_emb)

            x = self.decoder(
                tgt=input_emb, memory=memory, tgt_mask=self.mask, memory_mask=self.memory_mask
            )  # (B, T, n_emb)

        # 8. 输出头
        x = self.ln_f(x)
        x = self.head(x)  # (B, T, output_dim)
        return x


# 测试代码
def test_modified_transformer():
    print("=== 测试修改后的模型 ===")

    # 参数设置
    batch_size = 10
    horizon = 16
    input_dim = 16
    output_dim = 16
    cond_dim_image_encoded = 2048  # 编码后的图像特征
    cond_dim_traj = 9  # 单个时间步的轨迹维度
    cond_time_steps = 16  # T'

    # 创建模型
    transformer = VideoTransformerForDiffusion(
        input_dim=input_dim,
        output_dim=output_dim,
        horizon=horizon,
        cond_dim_video_encoded=cond_dim_image_encoded,
        cond_dim_traj=cond_dim_traj,
        cond_time_steps=cond_time_steps,
        causal_attn=True,
        n_cond_layers=4,
        preserve_traj_temporal=True,  # 保留轨迹时间结构
    )

    # 创建测试数据
    timestep = torch.tensor(0)
    sample = torch.randn(batch_size, horizon, input_dim)

    # 条件数据：(B, cond_dim_image_encoded + T' * cond_dim_traj)
    # total_cond_dim = cond_dim_image_encoded + cond_time_steps * cond_dim_traj
    video_cond = torch.randn(batch_size, cond_dim_image_encoded)
    traj_cond = torch.randn(batch_size, cond_time_steps, cond_dim_traj)
    # cond = torch.randn(batch_size, total_cond_dim)

    print(f"输入形状：{sample.shape}")
    print(f"图像条件形状：{video_cond.shape}")
    print(f"轨迹条件形状：{traj_cond.shape}")
    print(f"图像特征维度：{cond_dim_image_encoded}")
    print(f"轨迹数据维度：{cond_time_steps} x {cond_dim_traj} = {cond_time_steps * cond_dim_traj}")

    # 前向传播
    output = transformer(sample, timestep, video_cond, traj_cond)
    print(f"输出形状：{output.shape}")

    # 测试不保留时间结构的版本
    print("\n=== 测试展平轨迹版本 ===")
    transformer_flat = VideoTransformerForDiffusion(
        input_dim=input_dim,
        output_dim=output_dim,
        horizon=horizon,
        cond_dim_video_encoded=cond_dim_image_encoded,
        cond_dim_traj=cond_dim_traj,
        cond_time_steps=cond_time_steps,
        causal_attn=True,
        n_cond_layers=4,
        preserve_traj_temporal=False,  # 不保留轨迹时间结构
    )

    output_flat = transformer_flat(sample, timestep, video_cond, traj_cond)
    print(f"展平版本输出形状：{output_flat.shape}")

    print("\n测试完成！")


if __name__ == "__main__":
    test_modified_transformer()
