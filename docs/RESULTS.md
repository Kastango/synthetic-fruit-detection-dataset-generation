# Resultados dos 42 treinamentos

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

Os exemplos de `synthetic-2x` e `synthetic-3x` também estão incluídos acima. Reproduza com
`scripts/render_detection_examples.py` (troca a imagem/modelo editando as
constantes no topo do arquivo).

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
