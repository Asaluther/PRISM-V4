/* make_slides — PRISM v6.0 论文 PPT（17 页中文，pptxgenjs）
   配色与图表一致：HYB 蓝 / TF 红 / PCN 紫；深浅三明治；大数字母题 */
const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.layout = "LAYOUT_WIDE";
p.author = "PRISM";
p.title = "误差流与注意力的分工：PRISM 技术报告 v6.0";

const W = 13.33, H = 7.5, M = 0.55;
const DARK = "0F2440", LIGHT = "FFFFFF", PANEL = "F1F5F9";
const HYB = "1D4ED8", TF = "DC2626", PCN = "7C3AED";
const TXT = "1E293B", MUT = "64748B", INV = "E2E8F0";
const F = "Microsoft YaHei";
const bu = () => ({ code: "25B8", indent: 12, color: MUT });
const FIG = "results/figures/";
// 图宽高比（w/h）
const AR = { inverted_u: 1635 / 619, cross_scale: 1185 / 615, decouple: 1180 / 615,
             step50: 1185 / 662, pcn_diverge: 1185 / 615, multidraw: 1185 / 660 };

function base(dark) {
  const s = p.addSlide();
  s.background = { color: dark ? DARK : LIGHT };
  return s;
}
function title(s, t, sub, dark) {
  s.addText(t, { x: M, y: 0.38, w: W - 2 * M, h: 0.75, fontSize: 30, bold: true,
    fontFace: F, color: dark ? "FFFFFF" : TXT, margin: 0 });
  if (sub) s.addText(sub, { x: M, y: 1.12, w: W - 2 * M, h: 0.4, fontSize: 14,
    fontFace: F, color: dark ? INV : MUT, margin: 0 });
}
function fig(s, name, x, y, h) {
  const w = h * AR[name];
  s.addImage({ path: FIG + name + ".png", x, y, w, h });
  return w;
}
function src(s, t) {
  s.addText(t, { x: M, y: H - 0.42, w: W - 2 * M, h: 0.3, fontSize: 10.5,
    fontFace: F, color: MUT, margin: 0 });
}
function num(s, v, lbl, x, y, w, c, big) {
  s.addText(v, { x, y, w, h: 1.1, fontSize: big || 54, bold: true,
    fontFace: F, color: c, align: "center", margin: 0 });
  s.addText(lbl, { x, y: y + 1.02, w, h: 0.66, fontSize: 12.5,
    fontFace: F, color: MUT, align: "center", margin: 0 });
}

/* S1 封面 */
(() => { const s = base(true);
  s.addText("误差流与注意力的分工", { x: M, y: 2.0, w: W - 2 * M, h: 1.1,
    fontSize: 52, bold: true, fontFace: F, color: "FFFFFF", margin: 0 });
  s.addText("预测编码混合架构的预训练优势、自限适应与测量纪律",
    { x: M, y: 3.2, w: W - 2 * M, h: 0.6, fontSize: 22, fontFace: F, color: "93C5FD", margin: 0 });
  s.addText([
    { text: "PRISM 技术报告 v6.0  ·  2026-09-21", options: { breakLine: true } },
    { text: "单卡 RTX 4080 · 209 有效训练 run · github.com/Asaluther/PRISM-V4" },
  ], { x: M, y: 5.9, w: W - 2 * M, h: 0.9, fontSize: 15, fontFace: F,
    color: INV, margin: 0, paraSpaceAfter: 6 });
})();

/* S2 研究问题 */
(() => { const s = base(false);
  title(s, "研究问题：终端智能的两个区间", "北极星：终端私有智能 —— 低功耗、可在线学习、数据不出端");
  const cw = (W - 2 * M - 0.5) / 2;
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: M, y: 1.9, w: cw, h: 3.4,
    fill: { color: PANEL }, rectRadius: 0.08 });
  s.addText("数据充裕区 · 预训练", { x: M + 0.35, y: 2.2, w: cw - 0.7, h: 0.5,
    fontSize: 19, bold: true, fontFace: F, color: HYB, margin: 0 });
  s.addText([
    { text: "通用语言知识获取", options: { bullet: bu(), breakLine: true } },
    { text: "Transformer 的主场（101M 终判：调优 TF 领先纯 PCN 33.7%）", options: { bullet: bu(), breakLine: true } },
    { text: "大模型范式：能力不够 → 加规模 → 重训 → 算力电力上涨", options: { bullet: bu() } },
  ], { x: M + 0.35, y: 2.85, w: cw - 0.7, h: 2.2, fontSize: 15, fontFace: F,
    color: TXT, margin: 0, paraSpaceAfter: 10 });
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: M + cw + 0.5, y: 1.9, w: cw, h: 3.4,
    fill: { color: PANEL }, rectRadius: 0.08 });
  s.addText("数据稀缺区 · 个性化", { x: M + cw + 0.85, y: 2.2, w: cw - 0.7, h: 0.5,
    fontSize: 19, bold: true, fontFace: F, color: PCN, margin: 0 });
  s.addText([
    { text: "部署后的少样本适应（新用户/换域/冷启动）", options: { bullet: bu(), breakLine: true } },
    { text: "误差流架构的主场（冷启动 +58~74%，TF 全负）", options: { bullet: bu(), breakLine: true } },
    { text: "小模型 + 在线适应 = 对「大模型 + 反复重训」的替代路径", options: { bullet: bu() } },
  ], { x: M + cw + 0.85, y: 2.85, w: cw - 0.7, h: 2.2, fontSize: 15, fontFace: F,
    color: TXT, margin: 0, paraSpaceAfter: 10 });
  s.addText("核心问题：误差流类架构在两个区间各自的真实边界在哪里？能否在同一个模型内同时取得两者？",
    { x: M, y: 5.75, w: W - 2 * M, h: 0.8, fontSize: 18, bold: true, fontFace: F,
      color: TXT, align: "center", valign: "middle", margin: 0 });
  src(s, "答案：混合架构（12 层 Transformer 骨干 + 12 层 PCN 误差流头）——本报告主线");
})();

/* S3 平台与纪律 */
(() => { const s = base(false);
  title(s, "单卡研究平台与测量纪律", "纪律本身是贡献：七起系统性测量偏差全部被预设协议捕获");
  const stats = [["209", "有效训练 run\n（run_index 198 + 启智 11）"],
                 ["≤354M", "已测参数上界\n（21M → 354M）"],
                 ["55h", "GPU 总时长\n（330M 档全本地）"],
                 ["7 起", "测量偏差全捕获\n（含本研究自己的新主张）"]];
  stats.forEach((t, i) => num(s, t[0], t[1], M + i * (W - 2 * M) / 4, 1.85,
    (W - 2 * M) / 4 - 0.2, [HYB, PCN, HYB, TF][i]));
  s.addText([
    { text: "判据预注册：所有判决实验的通过阈值运行前写入脚本，事后不得调整；未命中如实报告", options: { bullet: bu(), breakLine: true } },
    { text: "对称调优条款：任何跨架构/跨规模对比前，双方超参各自在目标规模重扫", options: { bullet: bu(), breakLine: true } },
    { text: "同预算协议：一切对比固定 batch × seq × steps 的 token 总量", options: { bullet: bu(), breakLine: true } },
    { text: "第 7 起偏差（v6.0）：适应方法选择偏差 —— 以 full-FT 为唯一对照高估架构差异，由预注册判据触发主张重述", options: { bullet: bu(), breakLine: true } },
    { text: "主张：先怀疑测量，再相信架构", options: { bullet: bu() } },
  ], { x: M, y: 4.05, w: W - 2 * M, h: 2.6, fontSize: 15.5, fontFace: F,
    color: TXT, margin: 0, paraSpaceAfter: 12 });
  src(s, "数据：results/run_index.csv · 测量方法学详见报告 §3");
})();

/* S4 总览 */
(() => { const s = base(false);
  title(s, "四大发现（v6.0 定稿口径）", null);
  const cw = (W - 2 * M - 0.4) / 2, ch = 2.42, gx = M, gy = 1.55;
  const items = [
    ["① 倒 U 规律", "误差流优势随数据稀缺放大（至 +84%），参数轴峰位 ≈1025 tokens/param；一次带外预测成功", HYB],
    ["② 混合架构反超", "12TF+12PCN/96M：PPL 63.18±0.22，以更少参数反超纯 TF 36%（p≈0）；330M 外推 3/3 全胜", PCN],
    ["③ 适应的诚实边界", "真优势 = 零配置自限适应（27 格符号零翻转，跨规模保持）；TF+bitfit 可反超——但需方法选择", TF],
    ["④ 机制 ×2 转正", "主干锚定（16+8 消融三判读全中）；step50 达峰/回落 = 记忆与通用遗忘并行", HYB]];
  items.forEach((it, i) => {
    const x = gx + (i % 2) * (cw + 0.4), y = gy + Math.floor(i / 2) * (ch + 0.35);
    s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y, w: cw, h: ch,
      fill: { color: PANEL }, rectRadius: 0.07 });
    s.addText(it[0], { x: x + 0.3, y: y + 0.22, w: cw - 0.6, h: 0.5, fontSize: 19,
      bold: true, fontFace: F, color: it[2], margin: 0 });
    s.addText(it[1], { x: x + 0.3, y: y + 0.82, w: cw - 0.6, h: ch - 1.05, fontSize: 14.5,
      fontFace: F, color: TXT, margin: 0 });
  });
  src(s, "范围声明：参数 21-354M、token 预算 ≤164M、WikiText-103 / TinyStories 两域");
})();

/* S5 倒U */
(() => { const s = base(false);
  title(s, "发现①：规模-数据比规律", "纯 PCN（no_gating）vs Transformer，21-63M，5-seed，带外预测成功");
  const w = fig(s, "inverted_u", M, 1.75, 3.6);
  s.addText([
    { text: "减数据轴：数据越稀缺优势越大（+84% 封顶），无左端回落", options: { bullet: bu(), breakLine: true } },
    { text: "参数轴：倒 U，峰位 ≈1025 tokens/param（WT +60.8%，全胜）", options: { bullet: bu(), breakLine: true } },
    { text: "带外预测：WT d512 实测 +37.7% ∈ 预测带 [+30%, +55%]", options: { bullet: bu(), breakLine: true } },
    { text: "附带：误差流的 seed 方差小 1-2 个量级（端侧不可挑 seed 的独立优势）", options: { bullet: bu() } },
  ], { x: M, y: 5.6, w: W - 2 * M, h: 1.5, fontSize: 15, fontFace: F,
    color: TXT, margin: 0, paraSpaceAfter: 8 });
  src(s, "数据：results/analysis_v2/scale_curve_with_ci.json（11 点，bootstrap 95% CI）");
})();

/* S6 101M 终判 */
(() => { const s = base(false);
  title(s, "101M 终判：纯 PCN 的规模天花板", "对称调优后优势反转——失败机制是动力学内禀失稳，不是容量不足");
  num(s, "33.7%", "调优 TF 领先调优 PCN\n（99.34±2.62 vs 149.95±0.48）", M, 1.9, 3.6, TF);
  s.addText([
    { text: "PCN@3e-4 早期轨迹全场最佳（step2000 PPL 291 vs TF 766），随后崩入 fp16 溢出死循环", options: { bullet: bu(), breakLine: true } },
    { text: "bf16 重跑逐位复现劣化并平安跨过溢出点 —— 失稳与数值精度无关", options: { bullet: bu(), breakLine: true } },
    { text: "结论：误差正反馈使「快速学习区间」与「自毁区间」重合 —— 纯 PCN 的规模天花板", options: { bullet: bu(), breakLine: true } },
    { text: "v5 勘误史：「+40.1% 优势」系 TF 未调参伪影，已按协议撤回（第 6 起偏差）", options: { bullet: bu() } },
  ], { x: M + 4.1, y: 1.9, w: W - 2 * M - 4.1, h: 3.6, fontSize: 15.5, fontFace: F,
    color: TXT, margin: 0, paraSpaceAfter: 14 });
  s.addText("天花板既然是「稳定压住了快速」—— 解法不是降速，而是给误差流一个稳定的主干",
    { x: M, y: 5.85, w: W - 2 * M, h: 0.7, fontSize: 17, bold: true, fontFace: F,
      color: HYB, align: "center", valign: "middle", margin: 0 });
  src(s, "数据：results/SCALE_090M_VALIDATION.md · V8 机制实验（报告 §4.6）");
})();

/* S7 混合 96M */
(() => { const s = base(false);
  title(s, "发现②：混合架构反超纯 Transformer", "12TF+12PCN / 96M · 3-seed · 配对 bootstrap");
  fig(s, "cross_scale", M, 1.8, 4.1);
  num(s, "63.18±0.22", "混合 PPL（3-seed）", M + 7.0, 1.9, 5.2, HYB, 44);
  num(s, "-36%", "vs 纯 TF（99.34±2.62）\n配对 CI [-37.7,-33.2]，p≈0", M + 7.0, 3.5, 5.2, HYB, 44);
  s.addText("以更少参数（96.0M vs 101.5M）在数据稀缺峰区（853 t/p）大幅反超 —— 纯 TF 吃不到误差流加速，纯 PCN 被稳定性压住，12+12 各取所长",
    { x: M + 7.0, y: 5.15, w: 5.3, h: 1.3, fontSize: 14.5, fontFace: F, color: TXT, margin: 0 });
  src(s, "数据：results/SEED_REPLICATION.md · results/v7/seed_replication.json");
})();

/* S8 330M */
(() => { const s = base(false);
  title(s, "330-354M 外推：优势保持，PCN 天花板实测", "d1024/24L · 20K 步 · 164M tokens ≈ 470 t/p · 3-seed + 双侧 lr 夹逼");
  fig(s, "pcn_diverge", M, 1.8, 4.0);
  const rx = M + 5.6, rw = W - 2 * M - 5.6;
  const rows = [["HYB 330M", "39.75±0.34（夹逼最优 37.05）", HYB],
                ["TF 354M", "48.78±11.67（坏 seed 62.24）", TF],
                ["PCN 307M", "峰值 147.77 → 中途软发散 222", PCN]];
  rows.forEach((r, i) => {
    s.addText(r[0], { x: rx, y: 1.9 + i * 0.62, w: 2.0, h: 0.5, fontSize: 16, bold: true,
      fontFace: F, color: r[2], margin: 0 });
    s.addText(r[1], { x: rx + 2.05, y: 1.9 + i * 0.62, w: rw - 2.05, h: 0.5, fontSize: 14,
      fontFace: F, color: TXT, margin: 0 });
  });
  s.addText([
    { text: "逐 seed 配对 3/3 全胜；调优对调优 +11.2%（参数少 6.7%）", options: { bullet: bu(), breakLine: true } },
    { text: "seed 方差比 34×（96M 时 12×）—— 稳定性不对称随规模放大", options: { bullet: bu(), breakLine: true } },
    { text: "lr 容忍度分化：HYB 骨干上漂（1e-3 最优），TF 上探即死（发散）", options: { bullet: bu() } },
  ], { x: rx, y: 4.0, w: rw, h: 2.0, fontSize: 14.5, fontFace: F, color: TXT,
    margin: 0, paraSpaceAfter: 10 });
  src(s, "数据：results/SCALE_330M_VALIDATION.md（v2，含配置失误透明修复记录）");
})();

/* S9 解耦 */
(() => { const s = base(false);
  title(s, "收窄归因解耦：规模 vs 数据比", "36.4% → 11.2% 的收窄同时动了两个变量 —— 谁主导？");
  fig(s, "decouple", M, 1.85, 3.9);
  num(s, "1.76×", "token 效率：HYB 半数据（84.0）\n胜 TF 全数据（99.3）", M + 6.2, 2.1, 5.8, HYB, 50);
  s.addText([
    { text: "数据比效应（固定 96M，853→470 t/p）：稀缺放大优势 +22pp", options: { bullet: bu(), breakLine: true } },
    { text: "规模效应（固定 470 t/p，96M→330M）：压缩 -47pp —— 收窄主导", options: { bullet: bu(), breakLine: true } },
    { text: "含义：优势最大区间 = 「小规模 + 数据稀缺」= 终端个人模型象限", options: { bullet: bu(), breakLine: true } },
    { text: "预测边界（如实标注）：500M+ 若压缩持续存在归零风险，500M 为判决性实验", options: { bullet: bu() } },
  ], { x: M + 6.2, y: 3.9, w: W - 2 * M - 6.2, h: 2.6, fontSize: 14, fontFace: F,
    color: TXT, margin: 0, paraSpaceAfter: 9 });
  src(s, "数据：run_tp470.sh 六训练 · results/wt_096m_hyb_tp470_s*/ · 报告 §4.7");
})();

/* S10 适应 */
(() => { const s = base(false);
  title(s, "发现③：适应优势跨规模保持（符号结构）", "全参数微调 27 格（3 架构 × 3 seed × 3 lr）零翻转");
  fig(s, "multidraw", M, 1.8, 3.9);
  s.addText([
    { text: "96M：PCN/HYB 全格为正，TF 全格为负（单发 +60.3±4.3 vs -57.5±10.5）", options: { bullet: bu(), breakLine: true } },
    { text: "330M 复测 + 10 用户多抽签：HYB 均值全正紧凑（正用户 27/30）vs TF 方差爆炸（σ 至 156）", options: { bullet: bu(), breakLine: true } },
    { text: "流式@5e-5：HYB +55~61% 且 held-out 双正；TF +19~26% 且 held-out 负（坏 seed -497%）", options: { bullet: bu(), breakLine: true } },
    { text: "ΔW 参数经济性保持（~140 vs ~230）", options: { bullet: bu() } },
  ], { x: M + 5.8, y: 1.9, w: W - 2 * M - 5.8, h: 4.0, fontSize: 14.5, fontFace: F,
    color: TXT, margin: 0, paraSpaceAfter: 12 });
  src(s, "数据：results/v7/adapt330_multidraw.json · adapt330.json · 报告 §7.7");
})();

/* S11 PEFT 重述 */
(() => { const s = base(false);
  title(s, "PEFT 对照与主张重述（第 7 起偏差）", "预注册判据触发：若 TF 的 PEFT 单发 ≥+30% 则适应主张需重述 —— 触发");
  const rows = [
    [{ text: "方法 @最优档", options: { bold: true, color: "FFFFFF", fill: { color: DARK } } },
     { text: "单发（TF / HYB）", options: { bold: true, color: "FFFFFF", fill: { color: DARK } } },
     { text: "流式@5e-5（TF / HYB）", options: { bold: true, color: "FFFFFF", fill: { color: DARK } } }],
    ["full-FT 5e-4", "-57.5±10.5 / +60.3±4.3", "全负 / +32.9±9"],
    ["LoRA r8 5e-4", "-29.8±13.5 / +57.1±6.0", "+73.0±1.5 / +77.7±2.7"],
    ["bitfit 1e-3", "+74.7±2.4 / +52.8±3.3", "+24.1±0.5 / +6.1±0.2"]];
  s.addTable(rows, { x: M, y: 1.85, w: W - 2 * M, colW: [3.2, 4.2, 4.8],
    fontSize: 14, fontFace: F, color: TXT, border: { pt: 0.5, color: "CBD5E1" },
    rowH: 0.5, valign: "middle", margin: 0.08 });
  s.addText([
    { text: "重述①：误差流提供「免方法选择、免步长调参的自限适应」——运维鲁棒性，终端的直接价值", options: { bullet: bu(), breakLine: true } },
    { text: "重述②：不是性能上限 —— TF+bitfit 单发 +74.7 可反超（但需要为每部署选对方法与配置）", options: { bullet: bu(), breakLine: true } },
    { text: "新发现：LoRA-TF 单发被 9 个 init 抽样钉死为负；bitfit 流式架构不对称（方法不可跨架构平移）", options: { bullet: bu() } },
  ], { x: M, y: 4.35, w: W - 2 * M, h: 2.3, fontSize: 15, fontFace: F, color: TXT,
    margin: 0, paraSpaceAfter: 12 });
  src(s, "数据：results/v7/peft_3seed.json（3-seed × 3 init 抽样）· PEFT_3SEED.md");
})();

/* S12 主干锚定 */
(() => { const s = base(false);
  title(s, "机制①：主干锚定（消融转正）", "假设 → 16+8@头2e-4 消融三判读全中 → 经验证机制");
  const cw = (W - 2 * M - 0.6) / 2;
  const rows = [
    [{ text: "判读", options: { bold: true, color: "FFFFFF", fill: { color: DARK } } },
     { text: "结果", options: { bold: true, color: "FFFFFF", fill: { color: DARK } } }],
    ["① 预训练稳定", "PPL 59.86 零 NaN（纯 PCN@2e-4 软发散）"],
    ["② 单发抬升", "+18.9% → +52.8%"],
    ["③ 阈值消失", "流式@1e-4 +3.1% → +59.1%"]];
  s.addTable(rows, { x: M, y: 1.9, w: cw, colW: [1.7, cw - 1.7], fontSize: 13.5,
    fontFace: F, color: TXT, border: { pt: 0.5, color: "CBD5E1" }, rowH: 0.52,
    valign: "middle", margin: 0.06 });
  num(s, "+72.0±2.0", "h2e4 终端默认 3-seed\n流式@5e-5（基线 +32.9±9.0，>4σ）", M + cw + 0.6, 1.95, cw + 0.0, HYB, 34);
  s.addText([
    { text: "TF 主干阻断误差正反馈的自毁路径 ——「快速=自毁」耦合被部分解耦", options: { bullet: bu(), breakLine: true } },
    { text: "当初设想的 qk-norm 稳定化路线不再必需", options: { bullet: bu(), breakLine: true } },
    { text: "部署地图：通用 12+12@头1e-4；终端在线学习默认 h2e4（头2e-4，阈值移动三 seed 全消）", options: { bullet: bu(), breakLine: true } },
    { text: "16+8@2e-4 三 seed 反转定性：流式专精（预训练 59.5 最好但单发不稳）", options: { bullet: bu() } },
  ], { x: M, y: 4.35, w: W - 2 * M, h: 2.3, fontSize: 14.5, fontFace: F, color: TXT,
    margin: 0, paraSpaceAfter: 11 });
  src(s, "数据：results/v7/{abl16_8_h2e4,h2e4_replication,abl16_8_3seed}.json · PHASE2_SENSITIVITY.md");
})();

/* S13 step50 */
(() => { const s = base(false);
  title(s, "机制②：step50 达峰/回落的分解", "终端预算「50 步」的科学根据");
  fig(s, "step50", M, 1.8, 3.85);
  s.addText([
    { text: "train PPL 单调背至 5-9：6 条序列被完全记忆（排除动力学失稳）", options: { bullet: bu(), breakLine: true } },
    { text: "test 峰值后回升，预训练域 PPL 同步恶化 6-47×：记忆与通用遗忘并行", options: { bullet: bu(), breakLine: true } },
    { text: "峰值高度 = 用户属性（同用户跨 lr 差 <4pp）；峰值步数 = lr 属性（降 lr 只拉伸时间轴）", options: { bullet: bu(), breakLine: true } },
    { text: "遗忘从 step10 即与适应并行 —— 预算是赛跑的止点，不是「够用」", options: { bullet: bu() } },
  ], { x: M + 5.6, y: 1.9, w: W - 2 * M - 5.6, h: 3.8, fontSize: 14.5, fontFace: F,
    color: TXT, margin: 0, paraSpaceAfter: 13 });
  num(s, "20s → +85%", "96M 混合模型纯 CPU 冷启动\n（峰值口径；终点口径低估 25pp）", M + 5.6, 5.35, 5.6, HYB, 32);
  src(s, "数据：results/STEP50_MECHANISM.md（3 用户 × 3 lr × 每 10 步 × 四曲线）");
})();

/* S14 终端 */
(() => { const s = base(false);
  title(s, "端侧链路：全本地、零网络调用", "96M 混合模型 · 纯 CPU（8 线程 fp32，366MB）· 4/4 判据命中");
  const st = [["2240", "tok/s 推理\n（b1_s256，判据 ≥500）"],
              ["43.8", "tok/s 生成\n（判据 ≥10）"],
              ["20s", "冷启动达峰 +85%\n（预算固化 50 步）"],
              ["0", "网络调用 / 云依赖\n数据不出端"]];
  st.forEach((t, i) => num(s, t[0], t[1], M + i * (W - 2 * M) / 4, 1.95,
    (W - 2 * M) / 4 - 0.2, [HYB, PCN, HYB, TF][i]));
  s.addText("对照 21M 小模型（8.6K tok/s / 113 生成 / 22.7s→+55% / 80MB）：参数 4.6× 换速度保留 26%，仍远超终端底线",
    { x: M, y: 4.15, w: W - 2 * M, h: 0.6, fontSize: 15, fontFace: F, color: TXT,
      align: "center", valign: "middle", margin: 0 });
  s.addText("终端形态各零件已全部验证：混合检查点 + 冻结骨干 + 50 步预算 + int4 配方（v1，限于小模型）+ CUDA Graph 能耗（-28%）",
    { x: M, y: 5.1, w: W - 2 * M, h: 0.9, fontSize: 15.5, bold: true, fontFace: F,
      color: HYB, align: "center", valign: "middle", margin: 0 });
  src(s, "数据：results/PHASE3_EDGE.md · results/v7/phase3_edge_hyb.json");
})();

/* S15 审稿回应 */
(() => { const s = base(false);
  title(s, "审稿回应状态（v4 版本意见）", "总体 7.5/10；高优先级建议全部执行");
  const rows = [
    [{ text: "审稿点", options: { bold: true, color: "FFFFFF", fill: { color: DARK } } },
     { text: "v6.0 回应", options: { bold: true, color: "FFFFFF", fill: { color: DARK } } },
     { text: "状态", options: { bold: true, color: "FFFFFF", fill: { color: DARK } } }],
    ["W1 规模外推不足", "330-354M × 3-seed + 夹逼 + 适应复测 + 收窄归因解耦（本地完成）", { text: "已解决", options: { bold: true, color: "166534" } }],
    ["W2 PEFT 基线缺失", "LoRA/bitfit × 双协议 × 3-seed × 3 init —— 超规格", { text: "已解决", options: { bold: true, color: "166534" } }],
    ["W2 RWKV/Mamba 代理", "编译限制未变，衰减代理维持", "挂账"],
    ["W3/W4 移植扩展、局部规则变体", "范围限定表述维持，实验列中期清单", "挂账（明确）"],
    ["W5 统计/修辞/章节乱序", "配对 CI+t、run_index 198 条、8/9 章节修复、版本历史", { text: "已解决", options: { bold: true, color: "166534" } }],
    ["Q1 配对检验 / Q5 中规模方案", "已补配对 bootstrap；Q5 旧「4080 不可行」判断被实测推翻并超越", { text: "已答/超越", options: { bold: true, color: "166534" } }]];
  s.addTable(rows, { x: M, y: 1.8, w: W - 2 * M, colW: [3.6, 6.4, 2.2], fontSize: 13,
    fontFace: F, color: TXT, border: { pt: 0.5, color: "CBD5E1" }, rowH: 0.55,
    valign: "middle", margin: 0.07 });
  src(s, "逐点明细：报告附录 D.3 · 独立版本说明 results/VERSION_NOTES_v6.md");
})();

/* S16 局限 */
(() => { const s = base(false);
  title(s, "局限与诚实边界", "每条主张绑定已测范围");
  s.addText([
    { text: "500M-1B 未测：B4 显示规模压缩主导收窄（-47pp），500M 为判决性实验（需云算力）", options: { bullet: bu(), breakLine: true } },
    { text: "330M 部分臂单 seed（HYB@1e-3、gating 对照、PCN 307M）；20K 步全部未饱和（预算内相对比较）", options: { bullet: bu(), breakLine: true } },
    { text: "适应协议为短流（8 批）/单发（6 序列）；真实长时程与真实用户数据未测", options: { bullet: bu(), breakLine: true } },
    { text: "复制优势不可外推到键值绑定（MQAR 否定性）；RWKV/Mamba 仍为代理", options: { bullet: bu(), breakLine: true } },
    { text: "冷启动用户抽签未播种：12+12 系 ±4 内无害，薄头/330M 档可达 33pp（已用多抽签协议对冲）", options: { bullet: bu(), breakLine: true } },
    { text: "量化/CUDA Graph/int4 结论继承 v1，限于 22M 小模型", options: { bullet: bu() } },
  ], { x: M, y: 1.8, w: W - 2 * M, h: 4.4, fontSize: 15.5, fontFace: F, color: TXT,
    margin: 0, paraSpaceAfter: 16 });
  src(s, "完整清单：报告 §9.2（v6.0 更新）");
})();

/* S17 结束 */
(() => { const s = base(true);
  s.addText("小模型 + 在线适应", { x: M, y: 2.3, w: W - 2 * M, h: 1.0, fontSize: 48,
    bold: true, fontFace: F, color: "FFFFFF", align: "center", margin: 0 });
  s.addText("对抗  大模型 + 反复重训", { x: M, y: 3.35, w: W - 2 * M, h: 0.8, fontSize: 30,
    fontFace: F, color: "93C5FD", align: "center", margin: 0 });
  s.addText("混合架构在「小规模 + 数据稀缺」象限优势最大 —— 恰是终端个人模型的所在",
    { x: M, y: 4.5, w: W - 2 * M, h: 0.6, fontSize: 17, fontFace: F, color: INV,
      align: "center", margin: 0 });
  s.addText("代码与数据开源：github.com/Asaluther/PRISM-V4  ·  PRISM 技术报告 v6.0（2026-09-21）",
    { x: M, y: 6.3, w: W - 2 * M, h: 0.5, fontSize: 13.5, fontFace: F, color: MUT,
      align: "center", margin: 0 });
})();

p.writeFile({ fileName: "results/PRISM_V6_SLIDES.pptx" }).then(() => console.log("done 17 slides"));
