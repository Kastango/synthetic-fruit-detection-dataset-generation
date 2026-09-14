# Resultados

A pergunta do experimento é se cena composta por computador substitui foto
anotada à mão no treino de um detector de poncã. Sete condições de treino,
duas arquiteturas e duas sementes fecham 28 execuções de 50 épocas em 960 px,
sem congelamento de camadas. O RT-DETR-L roda em seguida e entra nesta mesma
tabela.

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
| Pool sintético | `config_hash 6add1508ce3614ec4f064fdb`, manifesto `22827d15fef2baeb` |
| Manifesto `oranges_field` | `414470445dc9ce46` |
| Manifesto `manual_full_val` | `2be936523453bf5b` |
| Seleção | `c6fc1ebc0a157fc4`, 28 execuções |
| Ultralytics | 8.4.121 |

O teste foi aberto depois da seleção nos dois conjuntos, e a seleção olhou só
para a validação de origem de cada condição. Cada condição carrega a própria
impressão de manifesto: mudar um rótulo muda o identificador da execução, que
deixa de casar com o plano e é refeita.

## Coleta externa

| Detector | Condição | P | R | F1 | mAP@.50 | mAP@.75 | mAP@.50:.95 | MAE cont. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| yolov8s | manual-full | 0.586 | 0.384 | 0.464 | 0.381 | 0.096 | 0.149 | 5.9 |
| yolov8s | controlled | 0.052 | 0.005 | 0.009 | 0.003 | 0.002 | 0.002 | 500.0 |
| yolov8s | synthetic-1x | 0.635 | 0.240 | 0.346 | 0.274 | 0.091 | 0.121 | 10.4 |
| yolov8s | synthetic-2x | 0.523 | 0.325 | 0.401 | 0.318 | 0.113 | 0.144 | 9.4 |
| yolov8s | synthetic-3x | 0.525 | 0.314 | 0.393 | 0.316 | 0.110 | 0.142 | 9.9 |
| yolov8s | synthetic-5x | 0.581 | 0.353 | 0.439 | 0.365 | 0.128 | 0.164 | 8.8 |
| yolov8s | **synthetic-10x** | 0.578 | 0.365 | 0.448 | 0.379 | 0.133 | **0.172** | 7.9 |
| yolo26s | **manual-full** | 0.636 | 0.439 | 0.520 | 0.450 | 0.105 | **0.174** | 5.7 |
| yolo26s | controlled | 0.115 | 0.004 | 0.007 | 0.005 | 0.004 | 0.004 | 12.8 |
| yolo26s | synthetic-1x | 0.603 | 0.285 | 0.387 | 0.313 | 0.093 | 0.133 | 10.0 |
| yolo26s | synthetic-2x | 0.386 | 0.292 | 0.311 | 0.228 | 0.067 | 0.095 | 10.4 |
| yolo26s | synthetic-3x | 0.583 | 0.285 | 0.383 | 0.311 | 0.104 | 0.138 | 10.0 |
| yolo26s | synthetic-5x | 0.579 | 0.365 | 0.448 | 0.372 | 0.120 | 0.162 | 8.8 |
| yolo26s | synthetic-10x | 0.566 | 0.366 | 0.444 | 0.368 | 0.123 | 0.163 | 8.4 |

Negrito = maior média observada; não indica significância estatística.

## Validação da coleta própria

| Detector | Condição | P | R | F1 | mAP@.50 | mAP@.75 | mAP@.50:.95 | MAE cont. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| yolov8s | **manual-full** [val] | 0.937 | 0.806 | 0.867 | 0.902 | 0.615 | **0.554** | 2.7 |
| yolov8s | controlled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 500.0 |
| yolov8s | synthetic-1x | 0.782 | 0.440 | 0.563 | 0.524 | 0.310 | 0.301 | 9.6 |
| yolov8s | synthetic-2x | 0.797 | 0.519 | 0.629 | 0.598 | 0.349 | 0.337 | 10.1 |
| yolov8s | synthetic-3x | 0.755 | 0.550 | 0.636 | 0.616 | 0.344 | 0.344 | 8.7 |
| yolov8s | synthetic-5x | 0.817 | 0.516 | 0.633 | 0.607 | 0.351 | 0.344 | 9.1 |
| yolov8s | synthetic-10x | 0.808 | 0.547 | 0.652 | 0.638 | 0.374 | 0.359 | 8.0 |
| yolo26s | **manual-full** [val] | 0.905 | 0.811 | 0.855 | 0.890 | 0.611 | **0.546** | 2.5 |
| yolo26s | controlled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 17.8 |
| yolo26s | synthetic-1x | 0.806 | 0.547 | 0.652 | 0.624 | 0.361 | 0.349 | 8.4 |
| yolo26s | synthetic-2x | 0.781 | 0.550 | 0.645 | 0.618 | 0.359 | 0.350 | 8.2 |
| yolo26s | synthetic-3x | 0.750 | 0.545 | 0.631 | 0.606 | 0.352 | 0.341 | 8.8 |
| yolo26s | synthetic-5x | 0.807 | 0.557 | 0.659 | 0.639 | 0.379 | 0.365 | 7.9 |
| yolo26s | synthetic-10x | 0.796 | 0.558 | 0.656 | 0.642 | 0.379 | 0.364 | 7.1 |

`[val]` marca a condição que usou este mesmo conjunto para escolher checkpoint.

![mAP por volume sintético](figures/results/synthetic-volume-vs-map.svg)

Os dois painéis compartilham o eixo vertical de propósito: a distância entre
eles é o resultado, não um detalhe de formatação. Há também os gráficos de
[precision](figures/results/synthetic-volume-vs-precision.svg),
[recall](figures/results/synthetic-volume-vs-recall.svg) e
[F1](figures/results/synthetic-volume-vs-f1.svg).

## Leitura

**A vantagem do dado real é específica do domínio dele.** Na validação da
própria coleta, `manual-full` abre 0,19 de mAP@.50:.95 sobre a melhor condição
sintética — avaliada nas mesmas árvores, mesmo pomar, mesma câmera e mesmo
anotador do seu treino. Na coleta externa a vantagem some: 0,174 contra 0,163
no YOLO26s e 0,149 contra 0,172 no YOLOv8s, com os dois detectores discordando
de sinal. Fora do domínio de origem as duas fontes de treino empatam.

**Volume sintético satura perto de 5x.** De 1x para 5x o ganho externo é
+0,029 (YOLO26s) e +0,043 (YOLOv8s); de 5x para 10x, +0,001 e +0,008. Dobrar o
volume acima de 520 imagens não paga o custo de geração.

**A condição de controle colapsa, e é o que justifica o compositor.**
`controlled` treina em fruta fotografada em ambiente controlado, com caixa
vinda da máscara de segmentação, sem composição em copa: 0,004 e 0,002 de
mAP externo, zero na validação própria. No YOLOv8s o viés de contagem chega a
+487 caixas por imagem com precisão 0,05 — o detector pulveriza caixa em folha.
Fruta recortada sem cena não ensina a procurar fruta em cena.

**Todas as condições subestimam a contagem na coleta externa**, de −7,6 a
−10,4 frutas por imagem, contra −5,5 de `manual-full`. As sintéticas ainda
perdem fruta pequena e muito ocluída.

## Exemplos

Mesma cena, mesmo detector, uma coluna por condição de treino. Cena de
densidade mediana da coleta externa, 8 frutas no gabarito:

![Detecções por condição, cena mediana](figures/results/sheets/externo-cena-mediana.jpg)

`controlled` não devolve nada; as sintéticas sobem de 3 para 7 acertos com o
volume; `manual-full` fecha as 8. A [cena densa](figures/results/sheets/externo-cena-densa.jpg)
(41 frutas) e a [validação própria](figures/results/sheets/manual-full-val-yolo26s.jpg)
mostram o mesmo ordenamento.

As cenas que o compositor produz, com e sem caixa:

![Cenas sintéticas](figures/results/sheets/cenas-sinteticas.jpg)

A distribuição espacial das anotações de cada conjunto:

![Mapas de anotações](figures/results/sheets/mapas-de-anotacoes.jpg)

## Estabilidade e limites

São **duas sementes por condição**, o que serve para triagem e não para
promover uma condição sobre outra. A inversão de sinal entre YOLO26s e YOLOv8s
na coleta externa é exatamente o caso que pede quatro sementes.

O desvio entre as duas sementes, em mAP@.50:.95 na coleta externa:

| Condição | YOLO26s | YOLOv8s |
|---|---:|---:|
| manual-full | 0.0303 | 0.0196 |
| synthetic-1x | 0.0352 | 0.0241 |
| synthetic-2x | 0.0169 | 0.0232 |
| synthetic-3x | 0.0120 | 0.0006 |
| synthetic-5x | 0.0137 | 0.0002 |
| synthetic-10x | 0.0148 | 0.0007 |

**Volume compra estabilidade.** Com 104 cenas, a condição sintética é a menos
estável da tabela — 0,035 de desvio, acima das próprias 104 fotos reais. De 3x
em diante o YOLOv8s cai para 0,0006 e o YOLO26s se acomoda perto de 0,014.
`manual-full` fica entre as menos estáveis em qualquer volume: um pomar só, uma
câmera, 104 fotos.

Isso limita o que a tabela pode afirmar. A diferença entre `manual-full` e a
melhor sintética na coleta externa é 0,011 no YOLO26s e 0,023 no YOLOv8s,
contra desvios de 0,030 e 0,020 na própria `manual-full`. As duas fontes de
treino não são distinguíveis com duas sementes.

`oranges_field` é laranja doce na Sicília, não poncã em pomar brasileiro. Ela
mede transferência entre domínios de citro, não acurácia na tarefa final. E
suas estatísticas participaram da calibração da receita, então não é um teste
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
