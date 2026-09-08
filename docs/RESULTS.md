# Resultados dos detectores e exemplos de dados

Grade histórica, gráficos e exemplos abaixo. A rodada atual está em
[Primeira comparação pareada com YOLOv8s](#primeira-comparação-pareada-com-yolov8s).

**Rastreabilidade:** estas tabelas pertencem ao pool anterior mais denso.
A configuração atual usa 6% de cenas densas entre 60 e 110 objetos e não
reproduz estes números. A nova rodada de desenvolvimento usa somente YOLOv8s,
conforme [o protocolo do estúdio](GENERATOR_STUDIO.md).

Consolida 7 condições × 3 detectores × 2 sementes, com o protocolo de
[`configs/confirmatory.yaml`](../configs/confirmatory.yaml). Os nomes de pastas
`confirmatory` são mantidos para rastreabilidade. A análise atual é exploratória,
pois o desenvolvimento do gerador usou estatísticas dos conjuntos avaliados.

| Conjunto de avaliação | Composição | Papel e limite |
|---|---|---|
| CitDet | 119 imagens e 10.082 caixas do split oficial de teste do [CitDet](https://robotic-vision-lab.github.io/citdet/) | Coleta externa. Suas estatísticas orientaram ajustes de escala e densidade do gerador; portanto, não é um teste intocado pelo desenvolvimento. |
| `manual_full_val` | 26 imagens e 451 caixas da validação de `manual-full` | Avaliação local. Essas imagens selecionaram os checkpoints de `manual-full`, por isso essa condição não tem uma estimativa independente neste conjunto. |

As tabelas mostram médias das sementes 41 e 42. As médias arredondadas não
permitem calcular intervalos de confiança nem testar equivalência. A magnitude
do viés de seleção em `manual_full_val` não foi estimada. A diferença de coleta
e a seleção de checkpoints impedem atribuir as diferenças a uma única causa.

## Como reproduzir

Além do fluxo padrão do [`README.md`](../README.md#execução), os dois conjuntos
de avaliação foram registrados em `external_datasets` de
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

Por padrão, o último comando reproduz o gráfico a partir das médias
arredondadas versionadas em [`results-summary.json`](results-summary.json),
transcritas das tabelas publicadas no commit `9fc162f`. Para usar os resultados
locais com precisão completa, execute:

```bash
.venv/bin/python scripts/plot_confirmatory_results.py --results-dir artifacts/confirmatory
```

Os JSONs por execução ficam fora do Git. O snapshot versionado não contém
observações por semente e não permite reconstruir sua dispersão.
Relatórios completos com curvas de treino e CSVs detalhados:
`artifacts/confirmatory/RESULTS_citdet.md` e `RESULTS_manual_full_val.md`.

### Baixar os pesos treinados (sem retreinar)

Os 42 checkpoints (`best.pt`) da fase confirmatória e o `model_selection.json`
correspondente estão publicados como asset de release, validados por
SHA-256 em [`configs/pipeline.yaml`](../configs/pipeline.yaml) (`confirmatory_checkpoints`).
Num computador novo, depois de preparar os dados (`./run_pipeline.sh prepare
--device 0 --accept-data-terms`, que só baixa/organiza dados, sem treinar):

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

Release: https://github.com/Kastango/synthetic-fruit-detection-dataset-generation/releases/tag/confirmatory-checkpoints-v3

## Como mais dados sintéticos afetam o mAP

![mAP@0.50:0.95 por volume de dados sintéticos, CitDet e manual-full·val lado a lado](figures/results/synthetic-volume-vs-map.svg)

*Figura 1. Mesma escala vertical nos dois painéis, com origem em zero. Pontos e
linhas contínuas mostram médias dos volumes sintéticos; tracejados mostram
`manual-full` e pontilhados, `controlled`, para o mesmo detector. As linhas
conectam condições discretas e não constituem um ajuste de curva.*

`controlled` apresenta mAP entre 0,000 e 0,009 no CitDet, indicando baixa
transferência nas condições avaliadas. Os maiores valores sintéticos dos YOLOs
ocorrem em `2x`, mas os volumes seguintes oscilam. O RT-DETR chega a 0,212 em
`10x`, com queda de 0,184 em `2x` para 0,181 em `3x`; o crescimento não é
monotônico e estes pontos não estabelecem uma curva de saturação.

Os máximos sintéticos no CitDet excedem as médias de `manual-full` em 0,026,
0,007 e 0,051 para YOLOv8s, YOLO26s e RT-DETR-L. A escolha desses máximos entre
cinco volumes é posterior à avaliação. Não demonstra que toda condição
sintética supere a referência, nem que as diferenças sejam estatisticamente
sustentadas. Em `manual_full_val`, `manual-full` tem as maiores médias nos três
detectores, mas também usou esse conjunto na seleção dos checkpoints.

## Precision, recall e F1 por volume sintético

![Precision para YOLOv8s, YOLO26s e RT-DETR-L nos dois conjuntos](figures/results/synthetic-volume-vs-precision.svg)

![Recall para YOLOv8s, YOLO26s e RT-DETR-L nos dois conjuntos](figures/results/synthetic-volume-vs-recall.svg)

![F1 para YOLOv8s, YOLO26s e RT-DETR-L nos dois conjuntos](figures/results/synthetic-volume-vs-f1.svg)

Os três gráficos seguem a convenção do mAP: médias das sementes 41 e 42,
uma cor por detector, curvas sintéticas contínuas e referências `manual-full`
tracejadas e `controlled` pontilhadas. Cada métrica compartilha a escala entre
os dois conjuntos. F1 é a média registrada pelo avaliador, não a média
harmônica calculada a partir de precision e recall já arredondados.

## Tabela completa

`[val]` identifica checkpoints selecionados nas mesmas 26 imagens em que a
linha é avaliada. O destaque em negrito marca apenas a maior média observada
por detector e conjunto, sem teste de significância. P, R e F1 vêm do avaliador;
o limiar `conf=0.25` indicado nos exemplos abaixo não define o cálculo de AP.
MAE é o erro absoluto médio de contagem por imagem, não uma medida de safra.
A contagem usa o limiar de maior F1 da validação de origem de cada execução,
registrado na seleção, com fallback de 0,25 se o valor não estiver disponível.

### CitDet, coleta externa usada na calibração

| Detector | Condição | P | R | F1 | mAP@.50 | mAP@.75 | mAP@.50:.95 | Count MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| yolov8s | manual-full | 0.710 | 0.488 | 0.578 | 0.529 | 0.123 | 0.214 | 45.1 |
| yolov8s | controlled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 84.7 |
| yolov8s | synthetic-1x | 0.732 | 0.514 | 0.603 | 0.572 | 0.144 | 0.236 | 35.2 |
| yolov8s | **synthetic-2x** | 0.730 | 0.523 | 0.609 | 0.576 | 0.150 | **0.240** | 33.8 |
| yolov8s | synthetic-3x | 0.722 | 0.507 | 0.596 | 0.556 | 0.123 | 0.221 | 32.4 |
| yolov8s | synthetic-5x | 0.745 | 0.504 | 0.601 | 0.568 | 0.142 | 0.235 | 36.7 |
| yolov8s | synthetic-10x | 0.743 | 0.502 | 0.599 | 0.564 | 0.141 | 0.233 | 35.0 |
| rtdetr-l | manual-full | 0.578 | 0.414 | 0.479 | 0.423 | 0.078 | 0.161 | 54.8 |
| rtdetr-l | controlled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 84.7 |
| rtdetr-l | synthetic-1x | 0.613 | 0.419 | 0.498 | 0.429 | 0.099 | 0.172 | 45.3 |
| rtdetr-l | synthetic-2x | 0.634 | 0.480 | 0.547 | 0.494 | 0.091 | 0.184 | 27.2 |
| rtdetr-l | synthetic-3x | 0.628 | 0.468 | 0.536 | 0.476 | 0.093 | 0.181 | 42.0 |
| rtdetr-l | synthetic-5x | 0.677 | 0.499 | 0.574 | 0.524 | 0.118 | 0.207 | 33.6 |
| rtdetr-l | **synthetic-10x** | 0.706 | 0.515 | 0.595 | 0.544 | 0.116 | **0.212** | 26.9 |
| yolo26s | manual-full | 0.718 | 0.523 | 0.606 | 0.576 | 0.130 | 0.236 | 42.3 |
| yolo26s | controlled | 0.279 | 0.035 | 0.062 | 0.030 | 0.002 | 0.009 | 84.7 |
| yolo26s | synthetic-1x | 0.720 | 0.511 | 0.598 | 0.570 | 0.136 | 0.232 | 40.6 |
| yolo26s | **synthetic-2x** | 0.739 | 0.525 | 0.614 | 0.588 | 0.151 | **0.243** | 30.5 |
| yolo26s | synthetic-3x | 0.727 | 0.520 | 0.606 | 0.573 | 0.145 | 0.236 | 33.6 |
| yolo26s | synthetic-5x | 0.737 | 0.503 | 0.598 | 0.559 | 0.139 | 0.230 | 32.3 |
| yolo26s | synthetic-10x | 0.717 | 0.507 | 0.594 | 0.557 | 0.142 | 0.232 | 34.4 |

Negrito = maior média de mAP@.50:.95 entre as condições desse detector.

### Validação manual, avaliação local com reuso para seleção

| Detector | Condição | P | R | F1 | mAP@.50 | mAP@.75 | mAP@.50:.95 | Count MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| yolov8s | **manual-full** [val] | 0.925 | 0.814 | 0.866 | 0.896 | 0.602 | **0.549** | 2.3 |
| yolov8s | controlled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 17.3 |
| yolov8s | synthetic-1x | 0.782 | 0.595 | 0.674 | 0.681 | 0.433 | 0.400 | 4.6 |
| yolov8s | synthetic-2x | 0.778 | 0.618 | 0.689 | 0.693 | 0.417 | 0.400 | 4.7 |
| yolov8s | synthetic-3x | 0.769 | 0.595 | 0.671 | 0.676 | 0.396 | 0.390 | 4.5 |
| yolov8s | synthetic-5x | 0.810 | 0.594 | 0.685 | 0.693 | 0.391 | 0.394 | 4.2 |
| yolov8s | synthetic-10x | 0.781 | 0.606 | 0.682 | 0.681 | 0.421 | 0.396 | 4.9 |
| rtdetr-l | **manual-full** [val] | 0.823 | 0.775 | 0.798 | 0.797 | 0.518 | **0.469** | 2.2 |
| rtdetr-l | controlled | 0.004 | 0.032 | 0.006 | 0.001 | 0.000 | 0.000 | 17.3 |
| rtdetr-l | synthetic-1x | 0.715 | 0.507 | 0.591 | 0.550 | 0.333 | 0.313 | 5.0 |
| rtdetr-l | synthetic-2x | 0.695 | 0.476 | 0.560 | 0.533 | 0.299 | 0.287 | 5.6 |
| rtdetr-l | synthetic-3x | 0.771 | 0.500 | 0.607 | 0.573 | 0.345 | 0.327 | 6.9 |
| rtdetr-l | synthetic-5x | 0.707 | 0.527 | 0.603 | 0.576 | 0.342 | 0.330 | 6.8 |
| rtdetr-l | synthetic-10x | 0.784 | 0.499 | 0.610 | 0.570 | 0.347 | 0.330 | 6.5 |
| yolo26s | **manual-full** [val] | 0.911 | 0.815 | 0.860 | 0.894 | 0.618 | **0.552** | 2.4 |
| yolo26s | controlled | 0.139 | 0.041 | 0.063 | 0.021 | 0.001 | 0.007 | 17.3 |
| yolo26s | synthetic-1x | 0.734 | 0.635 | 0.680 | 0.679 | 0.419 | 0.397 | 4.9 |
| yolo26s | synthetic-2x | 0.807 | 0.640 | 0.714 | 0.706 | 0.434 | 0.412 | 4.8 |
| yolo26s | synthetic-3x | 0.823 | 0.614 | 0.703 | 0.696 | 0.437 | 0.412 | 4.5 |
| yolo26s | synthetic-5x | 0.765 | 0.650 | 0.703 | 0.699 | 0.425 | 0.409 | 4.3 |
| yolo26s | synthetic-10x | 0.783 | 0.642 | 0.705 | 0.701 | 0.443 | 0.413 | 4.0 |

Negrito = maior média de mAP@.50:.95 entre as condições desse detector.

## Exemplos de detecção

O gabarito usa caixas amarelas de 5 pixels com contorno preto, desenhadas
depois do redimensionamento para manter a legibilidade. As predições aparecem
em ciano. As coordenadas das anotações permanecem as originais.

### CitDet

Mesma imagem do teste CitDet (`ftp-6-60-43_fruit-drop-back-picture_1_2021-11-09-01-57-07`,
49 caixas de gabarito), avaliada com o checkpoint `yolo26s` (seed 41) de cada
condição, `conf ≥ 0.25`. É uma imagem ilustrativa, sem amostragem aleatória
documentada; não representa a distribuição de erros do conjunto. A imagem
completa abre ao clicar na miniatura.

| | |
|---|---|
| **Gabarito**<br><a href="figures/results/examples/ground-truth.jpg"><img src="figures/results/examples/ground-truth.jpg" width="360" alt="Detecções ou gabarito: ground-truth"></a> | **manual-full**<br><a href="figures/results/examples/manual-full.jpg"><img src="figures/results/examples/manual-full.jpg" width="360" alt="Detecções ou gabarito: manual-full"></a> |
| **controlled**<br><a href="figures/results/examples/controlled.jpg"><img src="figures/results/examples/controlled.jpg" width="360" alt="Detecções ou gabarito: controlled"></a> | **synthetic-1x**<br><a href="figures/results/examples/synthetic-1x.jpg"><img src="figures/results/examples/synthetic-1x.jpg" width="360" alt="Detecções ou gabarito: synthetic-1x"></a> |
| **synthetic-2x**<br><a href="figures/results/examples/synthetic-2x.jpg"><img src="figures/results/examples/synthetic-2x.jpg" width="360" alt="Detecções ou gabarito: synthetic-2x"></a> | **synthetic-3x**<br><a href="figures/results/examples/synthetic-3x.jpg"><img src="figures/results/examples/synthetic-3x.jpg" width="360" alt="Detecções ou gabarito: synthetic-3x"></a> |
| **synthetic-5x**<br><a href="figures/results/examples/synthetic-5x.jpg"><img src="figures/results/examples/synthetic-5x.jpg" width="360" alt="Detecções ou gabarito: synthetic-5x"></a> | **synthetic-10x**<br><a href="figures/results/examples/synthetic-10x.jpg"><img src="figures/results/examples/synthetic-10x.jpg" width="360" alt="Detecções ou gabarito: synthetic-10x"></a> |

### manual-full.val

Imagem `img_2021.jpg`, com **25 caixas de gabarito**, primeira em ordem
lexicográfica no split de validação. A seleção independe das predições.
Os três detectores usam os checkpoints históricos da semente 41, `conf=0.25`,
`imgsz=960` e `max_det=1000`. Esta cena ilustra diferenças de detecção;
contagens iguais não significam caixas corretas nem desempenho equivalente.

![Gabarito manual-full.val com 25 caixas amarelas](figures/results/examples/manual-full-val/yolov8s/ground-truth.jpg)

| Detector | manual-full | controlled | synthetic-3x |
|---|---|---|---|
| YOLOv8s | ![YOLOv8s manual-full, 25 detecções](figures/results/examples/manual-full-val/yolov8s/manual-full.jpg) | ![YOLOv8s controlled, nenhuma detecção](figures/results/examples/manual-full-val/yolov8s/controlled.jpg) | ![YOLOv8s synthetic-3x, 25 detecções](figures/results/examples/manual-full-val/yolov8s/synthetic-3x.jpg) |
| YOLO26s | ![YOLO26s manual-full, 26 detecções](figures/results/examples/manual-full-val/yolo26s/manual-full.jpg) | ![YOLO26s controlled, nenhuma detecção](figures/results/examples/manual-full-val/yolo26s/controlled.jpg) | ![YOLO26s synthetic-3x, 21 detecções](figures/results/examples/manual-full-val/yolo26s/synthetic-3x.jpg) |
| RT-DETR-L | ![RT-DETR-L manual-full, 41 detecções](figures/results/examples/manual-full-val/rtdetr-l/manual-full.jpg) | ![RT-DETR-L controlled, nenhuma detecção](figures/results/examples/manual-full-val/rtdetr-l/controlled.jpg) | ![RT-DETR-L synthetic-3x, 33 detecções](figures/results/examples/manual-full-val/rtdetr-l/synthetic-3x.jpg) |

Reprodução, sem novos treinos:

```bash
.venv/bin/python scripts/render_detection_examples.py
for model in yolov8s yolo26s rtdetr-l; do
  .venv/bin/python scripts/render_detection_examples.py \
    --dataset manual_full_val --model "$model" \
    --condition manual-full --condition controlled --condition synthetic-3x
done
```

Use `--image-stem` e `--seed` para escolher outra imagem e semente.
Os arquivos `provenance.json` junto das imagens registram hashes dos dados,
checkpoints, condições e contagens. O script verifica o hash de cada peso
contra a seleção histórica antes da inferência.

## Exemplos de dados sintéticos

Cenas existentes de `paired_reference`, geradas com semente raiz 42 e
`paired-v1`. São exemplos do gerador atual, separados do pool histórico dos
42 treinamentos acima. A seleção usa os quantis 25%, 50%, 75% e 97% da
contagem de caixas nas 390 cenas, com desempate pelo índice de geração;
não usa resultados de detector nem seleção estética. À direita, os rótulos
automáticos da mesma cena. As imagens sem caixas permitem inspecionar
problemas de inserção, escala e iluminação que as métricas não resumem.

| Cena composta | Gabarito automático |
|---|---|
| ![Cena 188, 7 frutas](figures/results/synthetic-examples/scene-1.jpg) | ![Cena 188, 7 caixas](figures/results/synthetic-examples/scene-1-boxes.jpg) |
| ![Cena 1, 16 frutas](figures/results/synthetic-examples/scene-2.jpg) | ![Cena 1, 16 caixas](figures/results/synthetic-examples/scene-2-boxes.jpg) |
| ![Cena 7, 25 frutas](figures/results/synthetic-examples/scene-3.jpg) | ![Cena 7, 25 caixas](figures/results/synthetic-examples/scene-3-boxes.jpg) |
| ![Cena 360, 73 frutas](figures/results/synthetic-examples/scene-4.jpg) | ![Cena 360, 73 caixas](figures/results/synthetic-examples/scene-4-boxes.jpg) |

Reproduza a exportação com `.venv/bin/python scripts/render_synthetic_examples.py`.
O dataset deve estar gerado conforme o [guia do estúdio](GENERATOR_STUDIO.md).
O [registro de origem](figures/results/synthetic-examples/provenance.json)
preserva sementes por cena, hashes, índices e a regra de seleção.

## Distribuição espacial das anotações

Área das caixas acumulada em coordenadas normalizadas, média por imagem;
branco = ausência, azul→amarelo = densidade crescente. Escala de cor
compartilhada entre todos os conjuntos.

| | | |
|---|---|---|
| **manual-full** (130 img / 2.093 caixas)<br><a href="figures/results/heatmaps/manual-full.png"><img src="figures/results/heatmaps/manual-full.png" width="240" alt="Cobertura média das caixas: manual-full"></a> | **controlled** (355 img / 127 caixas)<br><a href="figures/results/heatmaps/controlled.png"><img src="figures/results/heatmaps/controlled.png" width="240" alt="Cobertura média das caixas: controlled"></a> | **synthetic-1x** (130 img / 8.302 caixas)<br><a href="figures/results/heatmaps/synthetic-1x.png"><img src="figures/results/heatmaps/synthetic-1x.png" width="240" alt="Cobertura média das caixas: synthetic-1x"></a> |
| **synthetic-2x** (260 img / 18.084 caixas)<br><a href="figures/results/heatmaps/synthetic-2x.png"><img src="figures/results/heatmaps/synthetic-2x.png" width="240" alt="Cobertura média das caixas: synthetic-2x"></a> | **synthetic-3x** (390 img / 26.853 caixas)<br><a href="figures/results/heatmaps/synthetic-3x.png"><img src="figures/results/heatmaps/synthetic-3x.png" width="240" alt="Cobertura média das caixas: synthetic-3x"></a> | **synthetic-5x** (650 img / 44.284 caixas)<br><a href="figures/results/heatmaps/synthetic-5x.png"><img src="figures/results/heatmaps/synthetic-5x.png" width="240" alt="Cobertura média das caixas: synthetic-5x"></a> |
| **synthetic-10x** (1.300 img / 91.623 caixas)<br><a href="figures/results/heatmaps/synthetic-10x.png"><img src="figures/results/heatmaps/synthetic-10x.png" width="240" alt="Cobertura média das caixas: synthetic-10x"></a> | **CitDet**, teste (119 img / 10.082 caixas)<br><a href="figures/results/heatmaps/citdet.png"><img src="figures/results/heatmaps/citdet.png" width="240" alt="Cobertura média das caixas: citdet"></a> | **manual-full · val**, validação (26 img / 451 caixas)<br><a href="figures/results/heatmaps/manual_full_val.png"><img src="figures/results/heatmaps/manual_full_val.png" width="240" alt="Cobertura média das caixas: manual_full_val"></a> |

Os mapas somam a cobertura das caixas e dividem pelo número de imagens;
portanto, combinam posição, tamanho e quantidade de caixas. Não são mapas de
atenção do detector nem isolam a diversidade espacial. A concentração em
`controlled` é compatível com suas fotos de frutas isoladas, mas o mapa não
estabelece a causa do baixo desempenho. As figuras agregam treino e validação
nas condições de treinamento, conforme as contagens indicadas.

## O que os resultados permitem afirmar

1. A composição sintética apresentou maior transferência que `controlled`
   nas condições avaliadas. Essa comparação altera contexto, escala, densidade
   e quantidade de caixas ao mesmo tempo; não isola o efeito de DepthPro,
   sombras ou qualquer outra transformação.
2. Há volumes sintéticos com médias maiores que `manual-full` no CitDet para
   os três detectores. Essa observação precisa de confirmação em dados não
   usados no desenvolvimento e com a condição escolhida previamente.
3. A resposta ao volume depende do detector e dos hiperparâmetros usados.
   Mais imagens implicam mais passos por época e maior validação. Não se pode
   atribuir as diferenças apenas à arquitetura ou à diversidade sintética.
4. As menores médias de MAE no CitDet foram 30,5 para YOLO26s em `2x`, 32,4
   para YOLOv8s em `3x` e 26,9 para RT-DETR-L em `10x`. Os valores de
   `manual-full` foram 42,3, 45,1 e 54,8. São mínimos escolhidos após comparar
   condições; contagem por imagem não mede produção por árvore ou pomar, nem
   corrige frutos ocultos ou repetidos em diferentes vistas.

Para uma etapa confirmatória, é necessário congelar gerador, condições e
critério de seleção antes de acessar um novo conjunto reservado. A análise
precisa incluir variabilidade por semente e, quando possível, incerteza por
unidade de coleta, como árvore ou sessão, em vez de tratar fotos correlacionadas
como amostras independentes. Economia de anotação requer registrar horas de
coleta, preparação e revisão dos rótulos automáticos.

## Rastreabilidade

- SHA-256 da seleção de checkpoints: `99c19ee9733dc63914789c962d12e84b95f142e866355815ec95dca9b997023f`
- Checkpoints avaliados por teste: 42 (todos os treinamentos confirmatórios)
- O bloqueio da avaliação após `model_selection.json` não desfaz o uso prévio
  de estatísticas no gerador nem o reuso da validação manual para seleção.
- As médias versionadas registram a precisão das tabelas publicadas; auditoria
  por execução depende dos JSONs originais e dos manifestos do pool utilizado.

## Primeira comparação pareada com YOLOv8s

A candidata sem exposição independente e sem sombras projetadas foi
**rejeitada como melhoria**. O mAP@.50:.95 caiu nos dois cenários, nas duas
sementes. A distância de saturação também aumentou em ambas as coletas.
A interface mantém a receita atual como ponto de partida.

| Cenário | Receita atual, média | Candidata, média | Diferença |
|---|---:|---:|---:|
| Validação manual | 0,3787 | 0,3588 | −0,0199 |
| CitDet | 0,2101 | 0,1943 | −0,0158 |

| Receita | Semente de treino | Validação manual | CitDet |
|---|---:|---:|---:|
| Atual | 41 | 0,379996 | 0,214497 |
| Atual | 42 | 0,377453 | 0,205714 |
| Candidata | 41 | 0,367916 | 0,206601 |
| Candidata | 42 | 0,349688 | 0,181912 |

Cada braço usa 312 imagens de treino e 78 de validação, com 5.582 e 1.484
caixas respectivamente. As 390 imagens têm a mesma geometria e rótulos
entre os braços; apenas a aparência muda. Geração com semente 42,
`sampling.mode: paired-v1`, catálogo de 228 fundos e 127 recortes. Dois
mecanismos foram removidos em conjunto, reduzindo as folhas do YAML de 62
para 50; este resultado não isola a contribuição individual de cada um.

Treino com YOLOv8s, 50 épocas, tamanho 960, batch 8, SGD e pesos pré-treinados,
conforme [a configuração](../configs/similarity_yolov8.yaml). Os checkpoints
foram selecionados na validação sintética; os quatro foram congelados em
`artifacts/similarity_yolov8/paired_selection.json` antes da avaliação real.

Na comparação de aparência sobre o treino sintético inteiro, a distância
entre quantis de saturação subiu de 7,90 para 12,36 na coleta local e de 16,61
para 21,14 no CitDet (pontos percentuais da escala de saturação). A distância
de luminância caiu de 3,88 para 2,80 no local, mas subiu de 11,03 para 12,11
no CitDet. Logo, também não houve melhora conjunta das distribuições.
Essas medidas incluem o conteúdo das caixas, não apenas a casca das frutas.

Os dois conjuntos reais já participaram do desenvolvimento anterior. São
resultados exploratórios, não confirmação de generalização nem teste de
significância. Os números publicados da grade antiga pertencem a outro pool
e não substituem o controle pareado desta comparação.

O [snapshot com resultados, hashes e medidas de similaridade](studio-results.json)
permite consultar os números sem os pesos locais. Os artefatos completos
ficam em `artifacts/similarity_yolov8` e os checkpoints em
`runs/similarity_yolov8/training`.

A meta de melhorar o detector com menos parâmetros continua aberta. A
amostragem pareada, a rastreabilidade e a interface foram implementadas e
validadas; esta simplificação específica não foi adotada. A próxima hipótese
deve tratar um mecanismo por vez e partir de um problema visível nas cenas,
como contexto, posicionamento ou aparência local, preservando a densidade
escolhida no projeto.

## Ciclo de verossimilhança com YOLOv8s

<!-- realism-metrics:start -->
![YOLOv8s por receita e semente nos dois cenários](figures/results/realism/map.svg)

| Receita | CitDet 41 | CitDet 42 | Média | Local 41 | Local 42 | Média |
|---|---:|---:|---:|---:|---:|---:|
| manual-full | 0.223432 | 0.204814 | 0.214123 | 0.543716 | 0.554439 | 0.549077 |
| paired_essential | 0.206601 | 0.181912 | 0.194256 | 0.367916 | 0.349688 | 0.358802 |
| paired_reference | 0.214497 | 0.205714 | 0.210106 | 0.379996 | 0.377453 | 0.378725 |
| paired_canopy | 0.186433 | 0.216165 | 0.201299 | 0.304351 | 0.378800 | 0.341576 |
| paired_count_scale | 0.203593 | 0.213430 | 0.208512 | 0.351961 | 0.363115 | 0.357538 |
| paired_exposure | 0.208765 | 0.198986 | 0.203875 | 0.381753 | 0.360261 | 0.371007 |
| paired_highlights | 0.213855 | 0.201245 | 0.207550 | 0.399793 | 0.358221 | 0.379007 |
| paired_relief | 0.196127 | 0.196459 | 0.196293 | 0.382208 | 0.389005 | 0.385606 |
| paired_saturation | 0.213395 | 0.198733 | 0.206064 | 0.380718 | 0.382966 | 0.381842 |
| paired_shade | 0.186947 | 0.192797 | 0.189872 | 0.360671 | 0.373438 | 0.367055 |
| paired_sharpness | 0.196401 | 0.204688 | 0.200545 | 0.368514 | 0.360088 | 0.364301 |

Valores de mAP@.50:.95. Os pontos mostram as duas sementes de treino,
não intervalos de confiança. Todos os ciclos são exploratórios; ambos
os conjuntos reais já participaram do desenvolvimento. Os checkpoints
sintéticos foram selecionados somente na validação sintética.

O [snapshot completo](realism-results.json) registra métricas, similaridade
e hashes das fontes. Reproduza este bloco com
`.venv/bin/python scripts/report_realism.py` após a avaliação dos dois cenários.
<!-- realism-metrics:end -->

As três novas hipóteses mantêm 390 imagens, os 127 recortes, os mesmos fundos,
as sementes de treino 41/42 e 50 épocas. A geração usa semente 42, modo
pareado e a mistura original: 1–30 frutas, com probabilidade de 6% para
60–110. Foram realizadas 18 cenas densas neste pool; não se aumentou essa
probabilidade para perseguir mAP. O protocolo e a motivação anteriores aos
treinos estão no [guia do gerador](GENERATOR_STUDIO.md#experimento-reduzido).

| Receita | Mudança e referência | Inspeção das mesmas cenas |
|---|---|---|
| `paired_saturation` | Sobre `paired_reference`, perda de saturação ligada à exposição de 0,35 para 0,70. Mesmos 390 rótulos, byte a byte. | [Cenas 1 e 188](figures/results/realism/saturation.jpg) |
| `paired_relief` | Sobre `paired_saturation`, remove `bright_flatten_strength`. Mesmos rótulos; uma folha a menos no YAML. | [Cenas 1 e 188](figures/results/realism/relief.jpg) |
| `paired_count_scale` | Sobre `paired_reference`, escala proporcional a `sqrt(30 / contagem)` somente acima de 30 frutas. Nenhum novo ajuste numérico. | [Cenas densas 36 e 360](figures/results/realism/count-scale.jpg) |

As cenas 1 e 188 já integravam a inspeção anterior por quantis de contagem.
A cena 36 é a primeira cena densa em ordem de geração e a 360 já era o
exemplo denso publicado. Os índices foram definidos antes dos resultados
dos respectivos detectores; cada figura tem um JSON de proveniência junto.
As comparações mantêm a proporção e não retocam os resultados do compositor.

Na ablação de escala, 372 imagens esparsas e seus rótulos permaneceram
idênticos. As 18 cenas densas mantiveram fundos, recortes, sementes e contagens
sorteadas, com mudanças de tamanho e possíveis mudanças de inserção/oclusão.
São as mesmas 7.086 frutas solicitadas; as caixas finais passaram de 7.066
para 7.074, por menor perda na composição. A mediana do tamanho nas cenas de
treino com mais de 60 caixas caiu de 2,826% para 1,804%; no CitDet é 1,668%.
Isso aproxima a escala desse domínio, mas piora a distância global de tamanho
no domínio local: de 0,814 para 1,040 ponto percentual. No CitDet, essa
distância cai de 1,279 para 1,078. Não há melhora uniforme em todos os descritores.

A candidata de saturação não foi promovida: aproximou a distribuição de cor,
mas reduziu o mAP do CitDet nas duas sementes contra a receita de referência.
As cenas ainda revelam problemas de contexto, frutas sem ligação aparente
com galhos e oclusões fragmentadas. Uma distância menor de cor ou escala
não certifica que todo o dataset seja verossímil.

Para reproduzir o ciclo com os ativos preparados:

```bash
for recipe in paired_saturation paired_relief paired_count_scale; do
  .venv/bin/python scripts/generate_synthetic.py \
    --synthesis-config "configs/synthesis/$recipe.yaml" --workers 6
done
.venv/bin/python scripts/train_grid.py --config configs/realism_yolov8.yaml --device 0 --workers 4
.venv/bin/python scripts/evaluate_similarity.py --config configs/realism_yolov8.yaml --device 0
.venv/bin/python scripts/report_realism.py
.venv/bin/python scripts/render_realism_examples.py \
  --candidate paired_count_scale --scene 36 --scene 360 --output count-scale
```

O gerador recusa reutilizar diretórios de outra versão do código; use um
diretório novo para reconstruir dados antigos. Alterações somente no código
de proveniência podem preservar os pixels e ainda mudar o identificador do
experimento. Os hashes publicados registram a versão efetivamente avaliada.
