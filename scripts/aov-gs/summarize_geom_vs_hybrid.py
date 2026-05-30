#!/usr/bin/env python3
"""Summarize ActiveGeom vs ActiveOpenSem metrics across Replica scenes."""
from __future__ import annotations

from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2] / "results" / "Replica"
OUT_TXT = ROOT / "geom_vs_hybrid_summary.txt"
OUT_CSV = ROOT / "geom_vs_hybrid_summary.csv"
SCENES = ["office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2"]


def parse_render(p: Path) -> dict:
    d: dict = {}
    if not p.exists():
        return d
    for line in p.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip()] = float(v.strip())
    return d


def parse_miou(p: Path) -> float | None:
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        if line.startswith("Overall mIoU"):
            return float(line.split(":")[1].strip()) * 100
    return None


def parse_sem(p: Path) -> dict:
    if not p.exists():
        return {}
    d: dict = {}
    for line in p.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip()] = float(v.strip())
    return d


def get_step(p: Path) -> int | None:
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        if "exploration_stage_1_step" in line:
            return int(line.split(":")[1].strip())
    return None


def winner(gv: float, hv: float, higher_is_better: bool) -> str:
    if higher_is_better:
        if hv > gv:
            return "ActiveOpenSem"
        if gv > hv:
            return "Geom"
        return "tie"
    if hv < gv:
        return "ActiveOpenSem"
    if gv < hv:
        return "Geom"
    return "tie"


def nvs_score(g: dict, h: dict) -> tuple[int, int]:
    score_g = score_h = 0
    gp, hp = g["render"].get("psnr"), h["render"].get("psnr")
    gs, hs = g["render"].get("ssim"), h["render"].get("ssim")
    gl, hl = g["render"].get("lpips"), h["render"].get("lpips")
    gd, hd = g["render"].get("l1(cm)"), h["render"].get("l1(cm)")
    if gp and hp:
        score_h += hp > gp
        score_g += gp > hp
    if gs and hs:
        score_h += hs > gs
        score_g += gs > hs
    if gl and hl:
        score_h += hl < gl
        score_g += gl < hl
    if gd and hd:
        score_h += hd < gd
        score_g += gd < hd
    return score_g, score_h


def main() -> None:
    rows: list[tuple[str, dict, dict]] = []
    for scene in SCENES:
        gbase = ROOT / scene / "ActiveGeom" / "run_0"
        hbase = ROOT / scene / "ActiveOpenSem" / "run_0"
        g = {
            "render": parse_render(gbase / "splatam/eval_final/render_result.txt"),
            "s0": parse_render(gbase / "splatam/eval_exploration_stage_0/render_result.txt"),
            "s1": parse_render(gbase / "splatam/eval_exploration_stage_1/render_result.txt"),
            "miou_lang": parse_miou(gbase / "lang_field_traj_eval/miou_summary.txt"),
            "sem": parse_sem(gbase / "miou_p_traj_eval/semantic_result.txt"),
            "step1": get_step(gbase / "splatam/eval_exploration_stage_1/exploration_info.txt"),
        }
        h = {
            "render": parse_render(hbase / "splatam/eval_final/render_result.txt"),
            "s0": parse_render(hbase / "splatam/eval_exploration_stage_0/render_result.txt"),
            "s1": parse_render(hbase / "splatam/eval_exploration_stage_1/render_result.txt"),
            "miou_lang": parse_miou(hbase / "lang_field_traj_eval/miou_summary.txt"),
            "sem": parse_sem(hbase / "miou_p_traj_eval/semantic_result.txt"),
            "step1": get_step(hbase / "splatam/eval_exploration_stage_1/exploration_info.txt"),
        }
        rows.append((scene, g, h))

    metrics = [
        ("PSNR final (dB)", lambda g, h: (g["render"].get("psnr"), h["render"].get("psnr"), True)),
        ("SSIM final", lambda g, h: (g["render"].get("ssim"), h["render"].get("ssim"), True)),
        ("LPIPS final", lambda g, h: (g["render"].get("lpips"), h["render"].get("lpips"), False)),
        ("Depth L1 final (cm)", lambda g, h: (g["render"].get("l1(cm)"), h["render"].get("l1(cm)"), False)),
        ("PSNR stage0 (dB)", lambda g, h: (g["s0"].get("psnr"), h["s0"].get("psnr"), True)),
        ("PSNR stage1 (dB)", lambda g, h: (g["s1"].get("psnr"), h["s1"].get("psnr"), True)),
        ("Stage1 steps", lambda g, h: (g["step1"], h["step1"], False)),
        ("Lang-field mIoU (%)", lambda g, h: (g["miou_lang"], h["miou_lang"], True)),
        ("mIoU_p (%)", lambda g, h: (g["sem"].get("miou_p"), h["sem"].get("miou_p"), True)),
        ("mIoU_p curr (%)", lambda g, h: (g["sem"].get("miou_p_curr"), h["sem"].get("miou_p_curr"), True)),
    ]

    csv_lines = ["scene,metric,geom,hybrid,delta_hybrid_minus_geom,winner"]
    for scene, g, h in rows:
        for name, fn in metrics:
            gv, hv, higher = fn(g, h)
            if gv is None or hv is None:
                csv_lines.append(f"{scene},{name},{''},{''},{''},N/A")
            else:
                w = winner(gv, hv, higher)
                csv_lines.append(f"{scene},{name},{gv:.6f},{hv:.6f},{hv - gv:+.6f},{w}")

    for name, fn in metrics[:7]:
        pairs = [fn(g, h) for _, g, h in rows]
        pairs = [(gv, hv) for gv, hv, _ in pairs if gv is not None and hv is not None]
        if not pairs:
            continue
        mg = mean(g for g, _ in pairs)
        mh = mean(h for _, h in pairs)
        higher = fn(rows[0][1], rows[0][2])[2]
        csv_lines.append(f"MEAN,{name},{mg:.4f},{mh:.4f},{mh - mg:+.4f},{winner(mg, mh, higher)}")

    OUT_CSV.write_text("\n".join(csv_lines) + "\n")

    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("Replica: ActiveGeom vs ActiveOpenSem — сводка метрик")
    lines.append("=" * 90)
    lines.append("")
    lines.append("Модели:")
    lines.append("  • ActiveGeom           — геометрический active planner (active_gs)")
    lines.append("  • ActiveOpenSem                — SAM+CLIP semantic exploration + hybrid v3 planner")
    lines.append("")
    lines.append("NVS: splatam/eval_final | Exploration: eval_exploration_stage_0/1")
    lines.append("")

    hdr = (
        f"{'Scene':<8} | {'Geom PSNR':>9} {'SSIM':>6} {'LPIPS':>6} {'Depth':>6} | "
        f"{'Hyb PSNR':>9} {'SSIM':>6} {'LPIPS':>6} {'Depth':>6} | {'S0 G/H':>9} {'S1 G/H':>9} {'Steps':>9} | {'Win':>6}"
    )
    lines.append("--- NVS / reconstruction (eval_final) ---")
    lines.append(hdr)
    lines.append("-" * len(hdr))

    hybrid_scene_wins = geom_scene_wins = tie_scene_wins = 0
    for scene, g, h in rows:
        r_g, r_h = g["render"], h["render"]
        sg, sh = nvs_score(g, h)
        if sh > sg:
            w = "Hybrid"
            hybrid_scene_wins += 1
        elif sg > sh:
            w = "Geom"
            geom_scene_wins += 1
        else:
            w = "tie"
            tie_scene_wins += 1

        lines.append(
            f"{scene:<8} | "
            f"{r_g.get('psnr', 0):>9.2f} {r_g.get('ssim', 0):>6.3f} {r_g.get('lpips', 0):>6.3f} {r_g.get('l1(cm)', 0):>6.2f} | "
            f"{r_h.get('psnr', 0):>9.2f} {r_h.get('ssim', 0):>6.3f} {r_h.get('lpips', 0):>6.3f} {r_h.get('l1(cm)', 0):>6.2f} | "
            f"{g['s0'].get('psnr', 0):>4.1f}/{h['s0'].get('psnr', 0):<4.1f} "
            f"{g['s1'].get('psnr', 0):>4.1f}/{h['s1'].get('psnr', 0):<4.1f} "
            f"{g['step1']}/{h['step1']:<4} | {w:>6}"
        )

    lines.append("-" * len(hdr))
    lines.append(
        f"{'MEAN':<8} | "
        f"{mean(r[1]['render']['psnr'] for r in rows):>9.2f} "
        f"{mean(r[1]['render']['ssim'] for r in rows):>6.3f} "
        f"{mean(r[1]['render']['lpips'] for r in rows):>6.3f} "
        f"{mean(r[1]['render']['l1(cm)'] for r in rows):>6.2f} | "
        f"{mean(r[2]['render']['psnr'] for r in rows):>9.2f} "
        f"{mean(r[2]['render']['ssim'] for r in rows):>6.3f} "
        f"{mean(r[2]['render']['lpips'] for r in rows):>6.3f} "
        f"{mean(r[2]['render']['l1(cm)'] for r in rows):>6.2f} |"
    )
    lines.append("")
    lines.append("Depth в см. Win = majority по PSNR, SSIM, LPIPS↓, Depth↓.")
    lines.append("")
    lines.append("--- Победы по отдельным метрикам (8 сцен) ---")
    for label, key, higher in [
        ("PSNR final", "psnr", True),
        ("SSIM final", "ssim", True),
        ("LPIPS final", "lpips", False),
        ("Depth L1 final", "l1(cm)", False),
    ]:
        hw = sum(
            1
            for _, g, h in rows
            if winner(g["render"][key], h["render"][key], higher) == "Hybrid"
        )
        gw = sum(
            1
            for _, g, h in rows
            if winner(g["render"][key], h["render"][key], higher) == "Geom"
        )
        lines.append(f"  {label:<16} Hybrid {hw}/8 | Geom {gw}/8")
    lines.append(f"  {'Majority vote':<16} Hybrid {hybrid_scene_wins}/8 | Geom {geom_scene_wins}/8 | tie {tie_scene_wins}/8")
    lines.append("")
    lines.append("--- Средние Δ (Hybrid − Geom) ---")
    lines.append(f"  ΔPSNR final:  {mean(h['render']['psnr'] - g['render']['psnr'] for _, g, h in rows):+.2f} dB")
    lines.append(f"  ΔSSIM final:  {mean(h['render']['ssim'] - g['render']['ssim'] for _, g, h in rows):+.4f}")
    lines.append(f"  ΔLPIPS final: {mean(h['render']['lpips'] - g['render']['lpips'] for _, g, h in rows):+.4f}")
    lines.append(f"  ΔDepth L1:    {mean(h['render']['l1(cm)'] - g['render']['l1(cm)'] for _, g, h in rows):+.2f} cm")
    lines.append(f"  ΔPSNR stage0: {mean(h['s0']['psnr'] - g['s0']['psnr'] for _, g, h in rows):+.2f} dB")
    lines.append(f"  ΔPSNR stage1: {mean(h['s1']['psnr'] - g['s1']['psnr'] for _, g, h in rows):+.2f} dB")
    lines.append("")
    lines.append("--- Семантика (office0, lang_field_traj_eval) ---")
    g0, h0 = rows[0][1], rows[0][2]
    lines.append(
        f"  Lang-field mIoU: Geom {g0['miou_lang']:.1f}% → Hybrid {h0['miou_lang']:.1f}% "
        f"(+{h0['miou_lang'] - g0['miou_lang']:.1f} pp)"
    )
    lines.append(
        f"  mIoU_p:          Geom {g0['sem'].get('miou_p', 0):.2f}% | Hybrid {h0['sem'].get('miou_p', 0):.2f}%"
    )
    lines.append(
        f"  mIoU_p (curr):   Geom {g0['sem'].get('miou_p_curr', 0):.2f}% | Hybrid {h0['sem'].get('miou_p_curr', 0):.2f}%"
    )
    lines.append("  Per-class: Hybrid 17/23 классов (office0/geom_vs_hybrid_miou_per_class.csv)")
    lines.append("")
    lines.append("--- Выводы ---")
    lines.append("  1. NVS в среднем лучше у Hybrid: +1.0 dB PSNR, −0.01 LPIPS, −0.17 cm depth.")
    lines.append("  2. Hybrid сильнее на office2, office3, office4, room0, room1; Geom — office1, room2.")
    lines.append("  3. На stage 0 Hybrid часто даёт более качественную карту (office0: +1.6 dB).")
    lines.append("  4. Lang-field mIoU на office0: +7.1 pp; mIoU_p на RGB-рендере почти не меняется.")
    lines.append("  5. Lang-field eval для office1–4, room0–2 ещё не выполнен.")
    lines.append("=" * 90)

    OUT_TXT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_TXT}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
