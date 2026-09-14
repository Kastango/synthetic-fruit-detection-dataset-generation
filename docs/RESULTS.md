# Resultados

A pergunta do experimento é se cena composta por computador substitui foto
anotada à mão no treino de um detector de poncã. Sete condições de treino, três
arquiteturas e duas sementes fecham **42 execuções** de 50 épocas em 960 px, sem
congelamento de camadas.

A avaliação usa dois conjuntos com papéis distintos. `oranges_field` é coleta
externa: outro país, outra espécie de citro, outros sete celulares, nove
condições de luz, protocolo de anotação escrito por terceiros. `manual_full_val`
são as 26 fotos de validação da própria coleta — mesmo pomar, mesma câmera e
mesmo anotador do treino de `manual-full`, que também as usa para escolher
checkpoint. O primeiro mede transferência; o segundo mede desempenho dentro do
domínio e é otimista para `manual-full` por construção.

## Procedência

| Item | Valor |
|---|---|
| Pool sintético | `config_hash 65ab60add0904ccc5336fb59`, manifesto `96f19445076b69a3` |
| Catálogo de ativos | `f3ad9117f88b460eeb98fb7c` |
| Manifesto `oranges_field` | `414470445dc9ce46` |
| Manifesto `manual_full_val` | `2be936523453bf5b` |
| Seleção | `91d49504703da8ce`, 21 candidatos, 42 execuções |
| Ultralytics | 8.4.121 |

O teste foi aberto depois da seleção nos dois conjuntos, e a seleção olhou só
para a validação de origem de cada condição. Cada condição carrega a própria
impressão de manifesto: mudar um rótulo muda o identificador da execução, que
deixa de casar com o plano e é refeita.

## Coleta externa

![Ranking na coleta externa](figures/results/ranking-oranges-field.svg)

| Detector | Condição | P | R | F1 | mAP@.50 | mAP@.75 | mAP@.50:.95 | MAE cont. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| yolov8s | manual-full | 0.586 | 0.384 | 0.464 | 0.381 | 0.096 | 0.149 | 5.9 |
| yolov8s | controlled | 0.052 | 0.005 | 0.009 | 0.003 | 0.002 | 0.002 | 500.0 |
| yolov8s | synthetic-1x | 0.661 | 0.412 | 0.508 | 0.443 | 0.116 | 0.183 | 7.8 |
| yolov8s | synthetic-2x | 0.606 | 0.322 | 0.415 | 0.338 | 0.105 | 0.146 | 9.3 |
| yolov8s | synthetic-3x | 0.581 | 0.325 | 0.416 | 0.339 | 0.105 | 0.146 | 9.0 |
| yolov8s | **synthetic-5x** | 0.646 | 0.398 | 0.493 | 0.424 | 0.152 | **0.193** | 8.0 |
| yolov8s | synthetic-10x | 0.627 | 0.400 | 0.488 | 0.422 | 0.135 | 0.185 | 7.4 |
| rtdetr-l | manual-full | 0.625 | 0.394 | 0.483 | 0.390 | 0.089 | 0.149 | 6.3 |
| rtdetr-l | controlled | 0.073 | 0.010 | 0.018 | 0.008 | 0.005 | 0.005 | 12.8 |
| rtdetr-l | synthetic-1x | 0.662 | 0.367 | 0.470 | 0.385 | 0.132 | 0.171 | 7.6 |
| rtdetr-l | synthetic-2x | 0.642 | 0.382 | 0.479 | 0.396 | 0.122 | 0.170 | 7.9 |
| rtdetr-l | **synthetic-3x** | 0.666 | 0.422 | 0.515 | 0.451 | 0.146 | **0.197** | 7.4 |
| rtdetr-l | synthetic-5x | 0.596 | 0.367 | 0.454 | 0.384 | 0.127 | 0.170 | 9.9 |
| rtdetr-l | synthetic-10x | 0.583 | 0.369 | 0.452 | 0.380 | 0.122 | 0.165 | 9.9 |
| yolo26s | manual-full | 0.636 | 0.439 | 0.520 | 0.450 | 0.105 | 0.174 | 5.7 |
| yolo26s | controlled | 0.115 | 0.004 | 0.007 | 0.005 | 0.004 | 0.004 | 12.8 |
| yolo26s | synthetic-1x | 0.657 | 0.329 | 0.437 | 0.369 | 0.109 | 0.157 | 9.1 |
| yolo26s | synthetic-2x | 0.512 | 0.304 | 0.381 | 0.284 | 0.074 | 0.115 | 9.7 |
| yolo26s | synthetic-3x | 0.610 | 0.388 | 0.474 | 0.398 | 0.116 | 0.167 | 7.9 |
| yolo26s | **synthetic-5x** | 0.652 | 0.412 | 0.505 | 0.440 | 0.144 | **0.193** | 7.8 |
| yolo26s | synthetic-10x | 0.633 | 0.424 | 0.508 | 0.448 | 0.140 | 0.193 | 7.0 |

Negrito = maior média observada; não indica significância estatística.

## Validação da coleta própria

![Ranking na validação própria](figures/results/ranking-manual-full-val.svg)

| Detector | Condição | P | R | F1 | mAP@.50 | mAP@.75 | mAP@.50:.95 | MAE cont. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| yolov8s | **manual-full** [val] | 0.937 | 0.806 | 0.867 | 0.902 | 0.615 | **0.554** | 2.7 |
| yolov8s | controlled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 500.0 |
| yolov8s | synthetic-1x | 0.764 | 0.542 | 0.634 | 0.603 | 0.345 | 0.337 | 8.6 |
| yolov8s | synthetic-2x | 0.784 | 0.532 | 0.634 | 0.609 | 0.335 | 0.341 | 8.5 |
| yolov8s | synthetic-3x | 0.808 | 0.520 | 0.632 | 0.601 | 0.343 | 0.334 | 8.8 |
| yolov8s | synthetic-5x | 0.800 | 0.515 | 0.626 | 0.602 | 0.353 | 0.343 | 8.9 |
| yolov8s | synthetic-10x | 0.754 | 0.552 | 0.637 | 0.629 | 0.360 | 0.356 | 8.2 |
| rtdetr-l | **manual-full** [val] | 0.866 | 0.803 | 0.834 | 0.841 | 0.547 | **0.500** | 2.5 |
| rtdetr-l | controlled | 0.047 | 0.005 | 0.004 | 0.003 | 0.003 | 0.002 | 17.8 |
| rtdetr-l | synthetic-1x | 0.735 | 0.501 | 0.595 | 0.563 | 0.296 | 0.298 | 7.3 |
| rtdetr-l | synthetic-2x | 0.769 | 0.481 | 0.591 | 0.546 | 0.305 | 0.298 | 7.9 |
| rtdetr-l | synthetic-3x | 0.726 | 0.567 | 0.636 | 0.609 | 0.329 | 0.328 | 7.4 |
| rtdetr-l | synthetic-5x | 0.698 | 0.529 | 0.602 | 0.570 | 0.313 | 0.313 | 9.4 |
| rtdetr-l | synthetic-10x | 0.716 | 0.489 | 0.581 | 0.542 | 0.291 | 0.293 | 9.2 |
| yolo26s | **manual-full** [val] | 0.905 | 0.811 | 0.855 | 0.890 | 0.611 | **0.546** | 2.5 |
| yolo26s | controlled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 17.8 |
| yolo26s | synthetic-1x | 0.762 | 0.530 | 0.624 | 0.603 | 0.346 | 0.335 | 9.0 |
| yolo26s | synthetic-2x | 0.796 | 0.547 | 0.649 | 0.622 | 0.365 | 0.355 | 8.8 |
| yolo26s | synthetic-3x | 0.788 | 0.575 | 0.664 | 0.647 | 0.370 | 0.363 | 7.1 |
| yolo26s | synthetic-5x | 0.782 | 0.581 | 0.666 | 0.651 | 0.383 | 0.371 | 7.4 |
| yolo26s | synthetic-10x | 0.799 | 0.595 | 0.682 | 0.674 | 0.404 | 0.385 | 6.8 |

`[val]` marca a condição que usou este mesmo conjunto para escolher checkpoint.

## Curvas por volume

![mAP por volume sintético](figures/results/synthetic-volume-vs-map.svg)

Cada painel tem escala própria: as duas avaliações vivem em faixas diferentes de
mAP, e forçá-las ao mesmo eixo esconderia a forma das curvas. A faixa tracejada
é `manual-full` com o desvio entre suas duas sementes; a faixa colorida é o
mesmo desvio para cada curva sintética. `controlled` não aparece aqui — perto de
zero, ele achataria as curvas contra o topo; nos rankings acima, onde cada
condição tem linha própria, ele está.

![F1 por volume sintético](figures/results/synthetic-volume-vs-f1.svg)

As mesmas curvas para
[precision](figures/results/synthetic-volume-vs-precision.svg) e
[recall](figures/results/synthetic-volume-vs-recall.svg) mostram de onde vem o
F1: a precisão sintética encosta na real desde 1x, e o que o volume compra é
recall.

## Leitura

**Na coleta externa a receita sintética passa o gabarito manual, nas três
arquiteturas.** O melhor volume sintético abre +0,044 de mAP@.50:.95 sobre
`manual-full` no YOLOv8s (0,193 contra 0,149), +0,048 no RT-DETR-L (0,197 contra
0,149) e +0,020 no YOLO26s (0,193 contra 0,174). O sinal é o mesmo nos três, o
que não acontecia antes de a escala da fruta ser calibrada contra o pomar de
poncã. Mil e quarenta cenas compostas transferem melhor para outro país, outra
espécie e outras câmeras do que 104 fotos anotadas à mão de um pomar só.

**A vantagem do dado real é específica do domínio dele.** Na validação da
própria coleta o quadro inverte e por margem muito maior: `manual-full` abre
0,198 sobre a melhor sintética no YOLOv8s, 0,172 no RT-DETR-L e 0,161 no
YOLO26s — avaliada nas mesmas árvores, mesma câmera e mesmo anotador do seu
treino. As duas leituras não se contradizem: uma mede o domínio de origem, a
outra mede o que sai dele.

**Volume sintético satura entre 3x e 5x.** De 1x ao pico o ganho externo é
+0,010 (YOLOv8s, em 5x), +0,026 (RT-DETR-L, em 3x) e +0,036 (YOLO26s, em 5x).
De 5x para 10x o saldo é −0,008, −0,005 e 0,000. Dobrar o volume acima de 520
imagens não paga o custo de geração, e no RT-DETR-L chega a atrapalhar.

**A condição de controle colapsa, e é o que justifica o compositor.**
`controlled` treina em fruta fotografada em ambiente controlado, com caixa vinda
da máscara de segmentação, sem composição em copa: 0,002 a 0,005 de mAP externo,
e 0,000 a 0,002 na validação própria. No YOLOv8s o viés de contagem chega a +487
caixas por imagem com precisão 0,05 — o detector pulveriza caixa em folha. Fruta
recortada sem cena não ensina a procurar fruta em cena.

**Todas as condições subestimam a contagem na coleta externa**, de −6,8 a −9,9
frutas por imagem nas sintéticas, contra −5,5 a −6,1 em `manual-full`. As
sintéticas ainda perdem fruta pequena e muito ocluída — o mesmo déficit que
aparece como recall mais baixo.

## Exemplos

Quatro fotos do pomar de poncã ao lado de quatro cenas compostas, ambas com o
gabarito desenhado e escolhidas pela densidade mediana do seu conjunto:

![Pomar real ao lado de cena composta](figures/results/sheets/real-x-sintetico.jpg)

É esta comparação que a calibração de escala persegue: o centro de distância de
câmera por cena sai da distribuição de lado de caixa do pomar de poncã, não da
coleta externa, que é laranja doce fotografada mais de perto.

Mesma cena, mesmo detector, uma coluna por condição de treino. Cena de densidade
mediana da coleta externa, 8 frutas no gabarito:

![Detecções por condição, cena mediana](figures/results/sheets/externo-cena-mediana.jpg)

`controlled` não devolve nada; as sintéticas sobem com o volume; `manual-full`
fecha as 8. A [cena densa](figures/results/sheets/externo-cena-densa.jpg)
(41 frutas) mostra o mesmo ordenamento.

No pomar de poncã, onde `manual-full` joga em casa, dá para ver o que a
sintética ainda não alcança:

![Detecções na validação própria](figures/results/sheets/manual-full-val-yolo26s.jpg)

As cenas que o compositor produz, com e sem caixa:

![Cenas sintéticas](figures/results/sheets/cenas-sinteticas.jpg)

A distribuição espacial das anotações de cada conjunto:

![Mapas de anotações](figures/results/sheets/mapas-de-anotacoes.jpg)

## Estabilidade e limites

São **duas sementes por condição**, o que serve para triagem e não para promover
uma condição sobre outra. O desvio entre as duas sementes, em mAP@.50:.95 na
coleta externa:

| Condição | YOLOv8s | RT-DETR-L | YOLO26s |
|---|---:|---:|---:|
| manual-full | 0.0196 | 0.0356 | 0.0303 |
| synthetic-1x | 0.0199 | 0.0193 | 0.0087 |
| synthetic-2x | 0.0346 | 0.0219 | 0.0217 |
| synthetic-3x | 0.0100 | 0.0129 | 0.0051 |
| synthetic-5x | 0.0012 | 0.0040 | 0.0055 |
| synthetic-10x | 0.0089 | 0.0145 | 0.0047 |

**Volume compra estabilidade.** Em 5x o desvio cai para 0,001–0,006 nas três
arquiteturas, uma ordem de grandeza abaixo de `manual-full`, que fica entre as
condições menos estáveis em dois dos três detectores: um pomar só, uma câmera,
104 fotos.

Isso delimita o que a tabela pode afirmar. A vantagem sintética na coleta
externa (+0,020 a +0,048) supera o desvio de sementes de `manual-full` no
YOLOv8s e no RT-DETR-L, mas não no YOLO26s, onde +0,020 fica dentro dos 0,030 de
desvio da própria `manual-full`. O sinal é consistente nas três arquiteturas, o
que é mais forte do que qualquer painel isolado, mas duas sementes triam e não
promovem — a afirmação pede quatro.

`oranges_field` é laranja doce na Sicília, não poncã em pomar brasileiro. Ela
mede transferência entre domínios de citro, não acurácia na tarefa final. E suas
estatísticas participaram da calibração da receita, então não é um teste
confirmatório intocado — para isso seria preciso uma terceira coleta.

## Auditoria do gabarito

A validação automática confere rótulos, contagens, IDs congelados e hashes de
imagem e anotação contra o manifesto importado:

```bash
.venv/bin/python scripts/validate_data.py --stage real
```

Passa para as 130 imagens e 2.130 caixas, sem arquivo inválido, rótulo ausente,
repetição exata de imagem ou caixa duplicada. Isso não certifica que toda fruta
esteja anotada nem que cada caixa esteja visualmente correta — para isso existe
a revisão imagem a imagem:

```bash
.venv/bin/python scripts/audit_dataset.py \
  --images data/real_yolo_confirmatory/images/train \
  --labels data/real_yolo_confirmatory/labels/train \
  --output artifacts/auditoria/manual_full_train.jsonl --port 8770
```

Teclas 1–6 marcam `ok`, `faltando`, `caixa-frouxa`, `duplicada`,
`oclusao-extrema` e `nao-e-fruta`; Enter salva e avança. Com `--checkpoint`, a
fila começa pelas imagens em que detector e gabarito mais discordam, que é onde
o erro de anotação se concentra. Com `--resumo`, conta as marcas de um arquivo
já preenchido. A ferramenta registra julgamento e não muda coordenada.

## Reproduzir

```bash
.venv/bin/python scripts/train_grid.py --config configs/confirmatory.yaml --device 0
.venv/bin/python scripts/select_models.py --config configs/confirmatory.yaml
for d in oranges_field manual_full_val; do
  .venv/bin/python scripts/evaluate_test.py --config configs/confirmatory.yaml \
    --external-name "$d" --device 0 --unlock-test
done
.venv/bin/python scripts/plot_confirmatory_results.py
```

`train_grid.py` e `select_models.py` aceitam `--model` para rodar uma
arquitetura de cada vez e consolidar depois; `select_models.py` exige todas as
sementes de cada arquitetura que entrar. O protocolo, as receitas e a curadoria
dos conjuntos estão em [DATASETS.md](DATASETS.md).
