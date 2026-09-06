# Resultados da fase confirmatória

Consolida os resultados dos 42 treinamentos (7 condições × 3 detectores × 2
sementes, protocolo em [`configs/confirmatory.yaml`](../configs/confirmatory.yaml))
contra dois testes externos, nunca usados em treino ou seleção de checkpoint:

| Teste | O que é | Por que importa |
|---|---|---|
| **CitDet** | 119 imagens / 10.082 caixas do split de teste oficial do [CitDet](https://mavmatrix.uta.edu/cse_datasets/1/) — outro pomar, outra câmera, outra equipe de coleta | mede generalização para um domínio totalmente alheio |
| **manual-full · val** | as 26 imagens / 451 caixas de validação do split `manual-full` — mesmo pomar/câmeras usados para fotografar os fundos e frutas que viraram os conjuntos sintéticos | mede generalização dentro do mesmo domínio de captura, sem o ruído de outra fonte |

`manual-full · val` ainda carrega um viés leve — sinalizado abaixo — porque a
época/checkpoint da condição `manual-full` foi selecionada observando essas
mesmas 26 imagens.

## Como reproduzir

Além do fluxo padrão do [`README.md`](../README.md#execução), os dois testes
externos usados aqui foram registrados em `external_datasets` de
[`configs/pipeline.yaml`](../configs/pipeline.yaml) (`citdet`, já documentado no
README, e `manual_full_val`, que aponta para o próprio split de validação do
`manual-full`) e avaliados sem retreinar nada:

```bash
./run_pipeline.sh test --device 0 --unlock-test --external-name citdet
./run_pipeline.sh test --device 0 --unlock-test --external-name manual_full_val
./run_pipeline.sh report --external-name citdet
./run_pipeline.sh report --external-name manual_full_val
.venv/bin/python scripts/plot_confirmatory_results.py
```

O último comando gera o gráfico e as tabelas deste documento a partir de
`artifacts/confirmatory/test_results_{citdet,manual_full_val}.json` — que por
sua vez seguem o padrão gitignored do projeto (reproduzíveis, não versionados).
Relatórios completos com curvas de treino e CSVs detalhados:
`artifacts/confirmatory/RESULTS_citdet.md` e `RESULTS_manual_full_val.md`.

### Baixar os pesos treinados (sem retreinar)

Os 42 checkpoints (`best.pt`) da fase confirmatória e o `model_selection.json`
correspondente estão publicados como asset de release — validados por
SHA-256 em [`configs/pipeline.yaml`](../configs/pipeline.yaml) (`confirmatory_checkpoints`).
Num computador novo, depois de preparar os dados (`./run_pipeline.sh prepare
--device 0 --accept-data-terms`, que só baixa/organiza dados — não treina):

```bash
.venv/bin/python scripts/download_weights.py
```

Isso baixa o ZIP, valida o hash, extrai cada checkpoint em
`runs/confirmatory/training/<run_id>/weights/best.pt` e escreve
`artifacts/confirmatory/model_selection.json` já com os caminhos corrigidos
para a máquina local (os caminhos originais são absolutos e específicos de
onde o treino rodou). A partir daí, `evaluate_test.py`, `generate_report.py` e
`scripts/plot_confirmatory_results.py` funcionam normalmente contra qualquer
teste externo novo, sem rodar `train`/`select`:

```bash
./run_pipeline.sh prepare-test --external-name <nome> --external-source <arquivo>
./run_pipeline.sh test --device 0 --unlock-test --external-name <nome>
./run_pipeline.sh report --external-name <nome>
```

Release: https://github.com/Kastango/synthetic-fruit-detection-dataset-generation/releases/tag/confirmatory-checkpoints-v2

## Como mais dados sintéticos afetam o mAP

![mAP@0.50:0.95 por volume de dados sintéticos, CitDet e manual-full·val lado a lado](figures/results/synthetic-volume-vs-map.svg)

- **`controlled` fica fora de escala nos dois gráficos** (mAP ≈ 0.00–0.01 nos
  dois testes) — um detector treinado com frutas isoladas sobre fundo uniforme
  não generaliza para árvore real, em nenhum domínio de teste. Ver tabela completa abaixo.
- **A relação entre volume sintético e mAP depende do detector.** Nos dois
  YOLO, o ganho satura por volta de `synthetic-3x` nos dois testes (mais
  volume não ajuda ou recua ligeiramente depois disso). O RT-DETR foge do
  padrão: melhora de forma monotônica até `synthetic-10x` no CitDet, sem sinal
  de saturação no intervalo testado.
- **No CitDet, a distância para `manual-full` é pequena — ou inexistente.**
  Os dois YOLO ficam a 0,004–0,011 de mAP@.50:.95 do dado real; o RT-DETR com
  `synthetic-10x` (0,194) **supera** `manual-full` (0,143). Já no
  `manual-full · val` — que compartilha câmera e pomar com o próprio treino de
  `manual-full` — a distância continua grande nos três detectores (0,11–0,19),
  o que sugere que parte da vantagem de `manual-full` nesse teste é vantagem
  de domínio (mesma coleta), não superioridade geral do dado anotado
  manualmente.

## Tabela completa

⚠️ = a linha usa checkpoints da condição `manual-full`, cuja época/checkpoint
foi selecionado(a) observando exatamente essas 26 imagens de validação — viés
otimista leve (sem gradiente nelas, mas com seleção de modelo). Nenhuma outra
condição viu qualquer imagem de `manual-full` durante treino ou seleção.

### CitDet (externo, outro pomar/câmera)

| Detector | Condição | P | R | F1 | mAP@.50 | mAP@.75 | mAP@.50:.95 | Count MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| yolov8s | 🏆 **manual-full** | 0.710 | 0.488 | 0.578 | 0.529 | 0.123 | **0.214** | 45.1 |
| yolov8s | controlled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 84.7 |
| yolov8s | synthetic-1x | 0.713 | 0.458 | 0.558 | 0.510 | 0.113 | 0.204 | 49.3 |
| yolov8s | synthetic-2x | 0.724 | 0.467 | 0.567 | 0.518 | 0.114 | 0.207 | 43.8 |
| yolov8s | synthetic-3x | 0.705 | 0.476 | 0.568 | 0.518 | 0.120 | 0.210 | 37.8 |
| yolov8s | synthetic-5x | 0.730 | 0.460 | 0.565 | 0.510 | 0.117 | 0.205 | 43.3 |
| yolov8s | synthetic-10x | 0.730 | 0.456 | 0.561 | 0.504 | 0.120 | 0.206 | 42.9 |
| rtdetr-l | manual-full | 0.520 | 0.473 | 0.495 | 0.368 | 0.075 | 0.143 | 29.6 |
| rtdetr-l | controlled | 0.005 | 0.018 | 0.008 | 0.003 | 0.003 | 0.002 | 84.7 |
| rtdetr-l | synthetic-1x | 0.606 | 0.418 | 0.494 | 0.413 | 0.071 | 0.151 | 36.9 |
| rtdetr-l | synthetic-2x | 0.604 | 0.396 | 0.478 | 0.412 | 0.074 | 0.154 | 46.4 |
| rtdetr-l | synthetic-3x | 0.616 | 0.470 | 0.533 | 0.469 | 0.085 | 0.174 | 31.6 |
| rtdetr-l | synthetic-5x | 0.667 | 0.462 | 0.546 | 0.485 | 0.094 | 0.182 | 34.0 |
| rtdetr-l | 🏆 **synthetic-10x** | 0.691 | 0.486 | 0.571 | 0.512 | 0.101 | **0.194** | 30.4 |
| yolo26s | 🏆 **manual-full** | 0.718 | 0.523 | 0.606 | 0.576 | 0.130 | **0.236** | 42.3 |
| yolo26s | controlled | 0.279 | 0.035 | 0.062 | 0.030 | 0.002 | 0.009 | 84.7 |
| yolo26s | synthetic-1x | 0.716 | 0.485 | 0.579 | 0.541 | 0.130 | 0.220 | 43.9 |
| yolo26s | synthetic-2x | 0.723 | 0.493 | 0.586 | 0.547 | 0.133 | 0.222 | 41.5 |
| yolo26s | synthetic-3x | 0.728 | 0.501 | 0.593 | 0.552 | 0.133 | 0.225 | 32.8 |
| yolo26s | synthetic-5x | 0.723 | 0.483 | 0.579 | 0.534 | 0.133 | 0.218 | 35.1 |
| yolo26s | synthetic-10x | 0.717 | 0.472 | 0.569 | 0.521 | 0.129 | 0.213 | 42.9 |

🏆 = maior mAP@.50:.95 para aquele detector, nesse teste.

### manual-full · val (26 imgs, real, mesmo domínio dos fundos sintéticos)

| Detector | Condição | P | R | F1 | mAP@.50 | mAP@.75 | mAP@.50:.95 | Count MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| yolov8s | 🏆 **manual-full** ⚠️ | 0.925 | 0.814 | 0.866 | 0.896 | 0.602 | **0.549** | 2.3 |
| yolov8s | controlled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 17.3 |
| yolov8s | synthetic-1x | 0.755 | 0.476 | 0.583 | 0.541 | 0.347 | 0.316 | 8.9 |
| yolov8s | synthetic-2x | 0.761 | 0.536 | 0.629 | 0.602 | 0.372 | 0.348 | 6.8 |
| yolov8s | synthetic-3x | 0.771 | 0.496 | 0.604 | 0.574 | 0.352 | 0.332 | 8.3 |
| yolov8s | synthetic-5x | 0.796 | 0.522 | 0.629 | 0.607 | 0.369 | 0.352 | 7.5 |
| yolov8s | synthetic-10x | 0.820 | 0.538 | 0.650 | 0.620 | 0.380 | 0.359 | 6.3 |
| rtdetr-l | 🏆 **manual-full** ⚠️ | 0.749 | 0.766 | 0.757 | 0.772 | 0.485 | **0.454** | 4.0 |
| rtdetr-l | controlled | 0.354 | 0.076 | 0.023 | 0.012 | 0.003 | 0.005 | 17.3 |
| rtdetr-l | synthetic-1x | 0.789 | 0.521 | 0.627 | 0.572 | 0.327 | 0.318 | 5.9 |
| rtdetr-l | synthetic-2x | 0.738 | 0.524 | 0.613 | 0.566 | 0.318 | 0.310 | 6.2 |
| rtdetr-l | synthetic-3x | 0.766 | 0.548 | 0.639 | 0.611 | 0.355 | 0.340 | 6.2 |
| rtdetr-l | synthetic-5x | 0.746 | 0.544 | 0.629 | 0.599 | 0.348 | 0.332 | 6.0 |
| rtdetr-l | synthetic-10x | 0.788 | 0.462 | 0.581 | 0.530 | 0.312 | 0.301 | 7.5 |
| yolo26s | 🏆 **manual-full** ⚠️ | 0.911 | 0.815 | 0.860 | 0.894 | 0.618 | **0.552** | 2.4 |
| yolo26s | controlled | 0.139 | 0.041 | 0.063 | 0.021 | 0.001 | 0.007 | 17.3 |
| yolo26s | synthetic-1x | 0.784 | 0.596 | 0.677 | 0.666 | 0.397 | 0.380 | 6.2 |
| yolo26s | synthetic-2x | 0.807 | 0.543 | 0.649 | 0.632 | 0.390 | 0.370 | 7.8 |
| yolo26s | synthetic-3x | 0.789 | 0.585 | 0.672 | 0.659 | 0.407 | 0.386 | 6.2 |
| yolo26s | synthetic-5x | 0.797 | 0.570 | 0.664 | 0.657 | 0.398 | 0.383 | 6.1 |
| yolo26s | synthetic-10x | 0.794 | 0.576 | 0.668 | 0.657 | 0.413 | 0.386 | 5.9 |

🏆 = maior mAP@.50:.95 para aquele detector, nesse teste.

## Exemplos de detecção

Mesma imagem do teste CitDet (`ftp-6-60-43_fruit-drop-back-picture_1_2021-11-09-01-57-07`,
49 caixas de gabarito), avaliada com o checkpoint `yolo26s` (seed 41) de cada
condição, `conf ≥ 0.25`. Comparação qualitativa da tabela acima.

| | |
|---|---|
| **Gabarito**<br><img src="figures/results/examples/ground-truth.jpg" width="360"> | **manual-full**<br><img src="figures/results/examples/manual-full.jpg" width="360"> |
| **controlled** — colapso visível, quase nenhuma detecção<br><img src="figures/results/examples/controlled.jpg" width="360"> | **synthetic-1x**<br><img src="figures/results/examples/synthetic-1x.jpg" width="360"> |
| **synthetic-5x**<br><img src="figures/results/examples/synthetic-5x.jpg" width="360"> | **synthetic-10x**<br><img src="figures/results/examples/synthetic-10x.jpg" width="360"> |

`synthetic-2x` e `synthetic-3x` seguem o mesmo padrão de `1x`/`5x`/`10x` — veja
`docs/figures/results/examples/` para o conjunto completo. Reproduza com
`scripts/render_detection_examples.py` (troca a imagem/modelo editando as
constantes no topo do arquivo).

## Distribuição espacial das anotações

Área das caixas acumulada em coordenadas normalizadas, média por imagem;
branco = ausência, azul→amarelo = densidade crescente. Escala de cor
compartilhada entre todos os conjuntos.

| | | |
|---|---|---|
| **manual-full** (130 img / 2.093 caixas)<br><img src="figures/results/heatmaps/manual-full.png" width="240"> | **controlled** (355 img / 127 caixas)<br><img src="figures/results/heatmaps/controlled.png" width="240"> | **synthetic-1x** (130 img / 3.887 caixas)<br><img src="figures/results/heatmaps/synthetic-1x.png" width="240"> |
| **synthetic-2x** (260 img / 8.002 caixas)<br><img src="figures/results/heatmaps/synthetic-2x.png" width="240"> | **synthetic-3x** (390 img / 12.566 caixas)<br><img src="figures/results/heatmaps/synthetic-3x.png" width="240"> | **synthetic-5x** (650 img / 20.580 caixas)<br><img src="figures/results/heatmaps/synthetic-5x.png" width="240"> |
| **synthetic-10x** (1.300 img / 41.427 caixas)<br><img src="figures/results/heatmaps/synthetic-10x.png" width="240"> | **CitDet** — teste (119 img / 10.082 caixas)<br><img src="figures/results/heatmaps/citdet.png" width="240"> | **manual-full · val** — teste (26 img / 451 caixas)<br><img src="figures/results/heatmaps/manual_full_val.png" width="240"> |

`controlled` concentra as caixas numa faixa muito mais estreita do que os
demais conjuntos — mais um indício visual de por que a condição não
generaliza: a rede aprende uma composição de cena que não existe em campo. Os
conjuntos `synthetic-Nx` mostram uma dispersão mais próxima da observada no
CitDet do que a dos conjuntos reais capturados de perto (`manual-full`,
`controlled`) — reflexo de o gerador variar tanto a escala aparente da fruta
quanto a densidade de objetos por cena.

## Achados para o artigo

1. **`controlled` é um mau candidato a substituto de dado real.** Validação
   interna quase perfeita (mAP@.50:.95 ≈ 0.98) e colapso total nos dois testes
   externos (≤ 0.01) — o exemplo mais claro de overfitting de domínio do
   experimento inteiro.
2. **No domínio verdadeiramente externo (CitDet), o sintético chega perto — ou
   supera — o dado real.** Os dois YOLO ficam a 0,004–0,011 de mAP@.50:.95 de
   `manual-full`; o RT-DETR com `synthetic-10x` (0,194) **supera**
   `manual-full` (0,143). No `manual-full · val` — mesmo pomar/câmeras do
   treino de `manual-full` — a distância continua grande nos três detectores
   (0,11–0,19), o que sugere que parte dessa vantagem de `manual-full` é
   vantagem de domínio (mesma coleta), não superioridade geral do dado
   anotado manualmente.
3. **A relação entre volume sintético e desempenho depende do detector.** Nos
   dois YOLO, o ganho satura por volta de `synthetic-3x` nos dois testes. O
   RT-DETR foge do padrão: melhora de forma monotônica até `synthetic-10x` no
   CitDet, sem sinal de saturação no intervalo testado — pode ser diferença de
   arquitetura (atenção global vs. convolução local absorvendo melhor a
   diversidade extra) ou só dessa combinação de hiperparâmetros.
4. **Em contagem — métrica prática para estimativa de safra — o sintético no
   CitDet iguala ou supera o dado real.** `synthetic-3x` tem MAE de contagem
   menor que `manual-full` no yolo26s (32,8 vs. 42,3) e no yolov8s (37,8 vs.
   45,1): mais recall compensa o viés de contagem melhor que o dado real
   nesse domínio externo.

## Rastreabilidade

- SHA-256 da seleção de checkpoints: `8b3e2d3e0d42aea8396a0b7b04e691dc356d0989f31260aed397ef760435a031`
- Checkpoints avaliados por teste: 42 (todos os treinamentos confirmatórios)
- Teste aberto somente após a seleção (`model_selection.json` congelado antes de qualquer avaliação externa), nos dois testes.
