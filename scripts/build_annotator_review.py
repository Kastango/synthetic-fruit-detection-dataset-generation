#!/usr/bin/env python3
"""Painéis de evidências reais para revisão humana; não infere causa visual."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from analyze_misses import FONT, iou_matrix, read_boxes
from fruit_pipeline.common import ROOT, sha256_file

OUT = ROOT / "artifacts/annotator_review_cor_luz_2x"
CONF = 0.25
COLORS = {"gt": "#00d9ff", "pred": "#ffaf38", "low": "#ed84ff", "region": "#ffffff"}
TITLES = {
    "miss": "Sem caixa correspondente",
    "location": "Caixa próxima / IoU insuficiente",
    "low": "Candidata abaixo de 0,25",
    "fp": "Previsão sem anotação correspondente",
    "tp": "Acerto para comparação",
    "conflict": "Sem correspondência exclusiva",
}


def matches(gt, pred, scores):
    overlap = iou_matrix(pred, gt)
    used, pairs = set(), {}
    for j in np.argsort(-scores, kind="stable"):
        available = [i for i in range(len(gt)) if i not in used and overlap[j, i] >= 0.5]
        if available:
            i = max(available, key=lambda i: overlap[j, i])
            pairs[i] = int(j)
            used.add(i)
    return pairs


def font(size):
    return ImageFont.truetype(str(FONT), size)


def box_draw(draw, box, color, label=None, width=3):
    draw.rectangle(tuple(map(float, box)), outline=color, width=width)
    if label:
        x, y = max(0, box[0]), max(0, box[1] - 23)
        rect = draw.textbbox((x, y), label, font=font(18))
        draw.rectangle(rect, fill="#101820")
        draw.text((x, y), label, fill=color, font=font(18))


def crop_region(im, box):
    w, h = im.size
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    size = min(max(w, h), max(120, 4 * max(box[2] - box[0], box[3] - box[1])))
    x = max(0, min(w - size, cx - size / 2))
    y = max(0, min(h - size, cy - size / 2))
    return np.array([x, y, min(w, x + size), min(h, y + size)]).astype(int)


def project(box, region, scale):
    return (np.asarray(box) - np.tile(region[:2], 2)) * scale


def render_case(case, record, destination):
    im = Image.open(record["path"]).convert("RGB")
    target = np.array(case["box"])
    region = crop_region(im, target)
    raw = im.crop(tuple(region))
    scale = 480 / max(raw.size)
    raw = raw.resize(tuple(round(v * scale) for v in raw.size), Image.Resampling.LANCZOS)
    overlay = raw.copy()
    d = ImageDraw.Draw(overlay)
    for b in record["gt"]:
        if b[2] > region[0] and b[0] < region[2] and b[3] > region[1] and b[1] < region[3]:
            box_draw(d, project(b, region, scale), COLORS["gt"], width=2)
    for b, score in zip(record["pred"], record["scores"]):
        if b[2] > region[0] and b[0] < region[2] and b[3] > region[1] and b[1] < region[3]:
            box_draw(d, project(b, region, scale), COLORS["pred"], f"{score:.2f}", width=2)
    if case["kind"] == "low":
        box_draw(d, project(case["candidate_box"], region, scale), COLORS["low"],
                 f"{case['candidate_conf']:.3f} < 0.25", width=3)
    color = COLORS["pred"] if case["kind"] == "fp" else COLORS["gt"]
    target_label = f"ALVO {case['confidence']:.3f}" if case["kind"] == "fp" else "ALVO"
    box_draw(d, project(target, region, scale), color, target_label, width=4)
    context = im.copy()
    context.thumbnail((480, 480), Image.Resampling.LANCZOS)
    box_draw(ImageDraw.Draw(context), region * (context.width / im.width), "white", "RECORTE", width=3)
    panel = Image.new("RGB", (1520, 760), "#111820")
    d = ImageDraw.Draw(panel)
    d.text((20, 16), f"{case['id']} | {case['dataset']} | {TITLES[case['kind']]}", font=font(26), fill="white")
    for x, label, picture in [(20, "CONTEXTO: região do recorte", context),
                              (520, "RECORTE SEM MARCAÇÕES", raw), (1020, "ANOTAÇÕES + DETECTOR", overlay)]:
        d.text((x, 65), label, font=font(19), fill="#d0d8e0")
        panel.paste(picture, (x, 100))
    d.text((20, 600), "AZUL: anotação | LARANJA: previsão >= 0,25 | ROSA: candidata entre 0,05 e 0,25", font=font(21), fill="white")
    detail = f"Alvo: {target[2]-target[0]:.0f} x {target[3]-target[1]:.0f} px | "
    counterpart = "anotações" if case["kind"] == "fp" else "previsões >= 0,25"
    detail += f"IoU máximo com {counterpart}: {case['best_iou']:.3f}"
    d.text((20, 638), detail, font=font(21), fill="#c5d4e1")
    d.text((20, 677), "A causa visual e a correção da anotação ficam para sua avaliação.", font=font(20), fill="#c5d4e1")
    d.text((20, 719), Path(record["path"]).name[:125], font=font(16), fill="#9eafbf")
    panel.save(destination, quality=94)
    return overlay


def choose(candidates, count, used_images, selected):
    candidates = sorted(candidates, key=lambda c: (c["size"], c["image"], c["index"]))
    # Amostra tamanhos distintos, com no máximo dois alvos por fotografia.
    available = [c for c in candidates if (c["image"], c["source"], c["index"]) not in selected]
    picks = []
    for target_quantile in np.linspace(0.15, 0.85, count):
        if not available:
            break
        center = round(target_quantile * (len(available) - 1))
        order = sorted(range(len(available)), key=lambda i: abs(i - center))
        eligible = [i for i in order if used_images[available[i]["image"]] < 2]
        if not eligible:
            break
        c = available.pop(eligible[0])
        picks.append(c)
        used_images[c["image"]] += 1
        selected.add((c["image"], c["source"], c["index"]))
    return picks


def main():
    from ultralytics import YOLO
    OUT.mkdir(parents=True, exist_ok=True)
    spec = json.loads((ROOT / "artifacts/exp_cor_luz/test_specs_citdet/cand__yolov8s__s41__461e90bae6.json").read_text())
    assert sha256_file(Path(spec["checkpoint"])) == spec["checkpoint_sha256"]
    model = YOLO(spec["checkpoint"])
    report = dict(checkpoint=spec["checkpoint"], checkpoint_sha256=spec["checkpoint_sha256"],
                  training="receita original / synthetic-2x / semente de treino 41",
                  confidence=CONF, diagnostic_confidence=0.05, matching_iou=0.5, datasets={}, cases=[])
    for dataset, prefix in [("citdet", "C"), ("manual_full_val", "M")]:
        folder = ROOT / "data/external_tests" / dataset
        previous = json.loads((ROOT / f"artifacts/exp_cor_luz/test_results_{dataset}.json").read_text())
        if previous["external_manifest_sha256"] != sha256_file(folder / "manifest.json"):
            raise RuntimeError(f"{dataset}: dataset mudou; preserve a revisão antiga e reavalie os modelos antes de criar outra")
        images, records, candidates = sorted((folder / "images/test").glob("*.jpg")), {}, []
        totals = Counter()
        for path in images:
            im = Image.open(path)
            w, h = im.size
            gt = read_boxes(folder / "labels/test" / (path.stem + ".txt"), w, h)
            result = model.predict(str(path), conf=0.05, imgsz=960, max_det=1000, device="0", verbose=False)[0]
            all_boxes, all_scores = result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()
            high = all_scores >= CONF
            pred, scores = all_boxes[high], all_scores[high]
            low_boxes, low_scores = all_boxes[~high], all_scores[~high]
            pairs = matches(gt, pred, scores)
            overlap, low_overlap = iou_matrix(gt, pred), iou_matrix(gt, low_boxes)
            records[path.stem] = dict(path=str(path), gt=gt, pred=pred, scores=scores)
            totals.update(images=1, gt=len(gt), tp=len(pairs), fn=len(gt)-len(pairs), fp=len(pred)-len(pairs))
            for i, box in enumerate(gt):
                best = float(overlap[i].max()) if len(pred) else 0.0
                low_best = float(low_overlap[i].max()) if len(low_boxes) else 0.0
                kind = "tp" if i in pairs else "conflict" if best >= 0.5 else "low" if low_best >= 0.5 else "location" if best >= 0.1 else "miss"
                case = dict(image=path.stem, index=i, source="gt", kind=kind, box=box.tolist(),
                            best_iou=best, size=float(max(box[2]-box[0], box[3]-box[1])/w))
                if kind == "low":
                    j = int(low_overlap[i].argmax())
                    case.update(candidate_box=low_boxes[j].tolist(), candidate_conf=float(low_scores[j]),
                                candidate_iou=low_best)
                candidates.append(case)
            for j, box in enumerate(pred):
                if j not in pairs.values():
                    candidates.append(dict(image=path.stem, index=j, source="pred", kind="fp", box=box.tolist(),
                                           best_iou=float(overlap[:,j].max()) if len(gt) else 0.0,
                                           confidence=float(scores[j]), size=float(max(box[2]-box[0],box[3]-box[1])/w)))
        selected, used_images, cases = set(), Counter(), []
        for kind, count in [("miss", 2), ("location", 2), ("low", 2), ("fp", 3), ("tp", 3)]:
            cases.extend(choose([c for c in candidates if c["kind"] == kind], count, used_images, selected))
        if len(cases) < 12:
            cases.extend(choose([c for c in candidates if c["kind"] != "tp"], 12-len(cases), used_images, selected))
        overview = Image.new("RGB", (1260, 90 + 410 * ((len(cases)+2)//3)), "#111820")
        od = ImageDraw.Draw(overview)
        od.text((20, 10), f"{dataset} | receita original 2x | confiança 0,25", font=font(27), fill="white")
        od.text((20, 49), "Azul = anotação; laranja = detector; rosa = abaixo do limiar. ALVO identifica o caso.",font=font(19),fill="#d0d8e0")
        dest = OUT / dataset
        dest.mkdir(exist_ok=True)
        for i, case in enumerate(cases):
            case.update(id=f"{prefix}{i+1:02d}", dataset=dataset)
            case["panel"] = f"{dataset}/{case['id']}.jpg"
            overlay = render_case(case, records[case["image"]], OUT / case["panel"])
            overlay.thumbnail((400, 345))
            x, y = (i % 3)*420+10, (i//3)*410+90
            od.text((x,y), f"{case['id']} | {TITLES[case['kind']][:32]}", font=font(18), fill="white")
            overview.paste(overlay,(x,y+35))
            report["cases"].append(case)
        overview.save(OUT / f"overview_{dataset}.jpg", quality=94)
        # Preserva previsões numéricas e caminho da fotografia para auditoria.
        serial = {k:{kk:vv.tolist() if isinstance(vv,np.ndarray) else vv for kk,vv in v.items()} for k,v in records.items()}
        (dest / "predictions.json").write_text(json.dumps(serial))
        report["datasets"][dataset] = dict(totals=totals, available=Counter(c["kind"] for c in candidates), selected=len(cases))
        print(dataset, dict(totals), len(cases), "casos", flush=True)
    (OUT / "review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    cards = "\n".join(f'<article data-dataset="{c["dataset"]}"><h2>{c["id"]} — {TITLES[c["kind"]]}</h2>'
                        f'<a href="{c["panel"]}"><img loading="lazy" src="{c["panel"]}"></a>'
                        f'<textarea data-id="{c["id"]}" placeholder="Sua avaliação: dificuldade, anotação e alteração que faria"></textarea></article>' for c in report["cases"])
    (OUT / "index.html").write_text('''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Revisão do anotador</title>
<style>body{background:#111820;color:#edf3f8;font:18px system-ui;margin:24px auto;max-width:1520px;padding:0 16px}p{max-width:1000px;line-height:1.6}img{width:100%}article{margin:32px 0;border-top:1px solid #425364}textarea{box-sizing:border-box;width:100%;min-height:80px;font:17px system-ui;padding:12px}button,select{font:18px system-ui;padding:10px;margin-right:12px}nav{position:sticky;top:0;background:#111820;padding:12px 0}h2{font-size:24px}</style>
<h1>O que o detector acerta e perde</h1><p>Sua receita original, synthetic-2×, semente 41. Confiança 0,25 e IoU 0,50, com correspondência um a um. São 24 exemplos selecionados para diagnóstico, não uma amostra representativa das taxas de erro.</p>
<p>Azul: anotação. Laranja: previsão aceita. Rosa: candidata abaixo do limiar (não prova que reduzir o limiar resolveria o caso). Uma previsão sem par pode indicar falso positivo, duplicata ou anotação ausente. IoU insuficiente não prova que a caixa do detector esteja errada: confira o gabarito. O branco na imagem de contexto delimita o recorte.</p>
<p>O modelo foi escolhido pela semente, sem escolher o melhor resultado no teste. Não atribuí automaticamente causas como fruto verde, sombra ou oclusão. Você pode comentar pelos IDs ou preencher abaixo e exportar suas observações. Clique num painel para ampliar.</p>
<nav><select id="filter"><option value="all">Dois datasets</option value="citdet">CitDet</option><option value="manual_full_val">Manual val</option></select><button id="export">Exportar comentários</button></nav>''' + cards + '''
<script>const fields=[...document.querySelectorAll('textarea')];for(const f of fields){try{f.value=localStorage.getItem('review-cor-luz-'+f.dataset.id)||''}catch{}f.oninput=()=>{try{localStorage.setItem('review-cor-luz-'+f.dataset.id,f.value)}catch{}}}document.querySelector('#filter').onchange=e=>document.querySelectorAll('article').forEach(a=>a.hidden=e.target.value!=='all'&&a.dataset.dataset!==e.target.value);document.querySelector('#export').onclick=()=>{const data=Object.fromEntries(fields.map(f=>[f.dataset.id,f.value]));const u=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=u;a.download='avaliacao-anotador.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)}</script></html>''')


if __name__ == "__main__":
    main()
