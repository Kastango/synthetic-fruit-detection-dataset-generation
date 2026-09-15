# Resultados

A rodada publicada compara sete condições de treino em três detectores, com
sementes 41 e 42. As maiores médias sintéticas de mAP superam `manual-full` na
coleta externa. Na validação própria, `manual-full` lidera nas três arquiteturas.
O resultado indica que a utilidade dos dados sintéticos depende do domínio
avaliado e da métrica escolhida.

## Protocolo da rodada

São 42 execuções, com entrada de 960 pixels e duração configurada de até 50
épocas. Todas as camadas são ajustadas a partir de pesos pré-treinados. Cada
execução escolhe o checkpoint pela validação de sua condição. As fontes,
partições e configurações estão no [README](../README.md#dados-e-protocolo).

`manual_full_val` reúne as 26 fotos da validação própria, também usadas na
seleção de checkpoint de `manual-full`. `oranges_field` reúne 1.243 recortes
de laranja doce e mede transferência para outro domínio. Suas estatísticas
e seu protocolo de anotação participaram do desenvolvimento do gerador.
Esses dois usos delimitam a comparação como exploratória.

Os identificadores da rodada publicada são:

| Item | Valor |
|---|---|
| Pool sintético | `config_hash 65ab60add0904ccc5336fb59`, manifesto `96f19445076b69a3` |
| Catálogo de ativos | `f3ad9117f88b460eeb98fb7c`, 228 fundos e 127 recortes |
| Manifesto `oranges_field` | `414470445dc9ce46` |
| Manifesto `manual_full_val` | `2be936523453bf5b` |
| Seleção | `91d49504703da8ce`, 21 candidatos, 42 execuções |
| Ultralytics | 8.4.121 |

O manifesto de cada condição participa do identificador das execuções. A
configuração de síntese tem o hash indicado acima. Os registros dos exemplos
associam cada imagem aos checkpoints e à seleção usados na avaliação.

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

P é precision e R é recall. As métricas são médias das duas sementes. P e R
usam o ponto escolhido pelo avaliador em cada conjunto. O F1 é calculado por
execução antes da média. O mAP@.50:.95 resume AP em dez limiares de IoU, de
0,50 a 0,95. IoU é a razão entre a interseção e a união de duas caixas.

MAE cont. é o erro absoluto médio de contagem por imagem. A contagem usa o
limiar de melhor F1 da validação de origem de cada execução. O negrito marca a
maior média de mAP@.50:.95 por detector, calculada antes do arredondamento.

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

## Volume e desempenho

![mAP por volume sintético](figures/results/synthetic-volume-vs-map.svg)

Cada painel tem escala própria. A linha tracejada representa `manual-full`;
as faixas mostram o desvio-padrão entre as duas sementes. Os pontos das curvas
representam os cinco volumes sintéticos.

As maiores médias externas aparecem em 5x nos YOLOs e em 3x no RT-DETR-L.
Em mAP@.50:.95, são 0,193 contra 0,149 no YOLOv8s, 0,197 contra 0,149 no
RT-DETR-L e 0,193 contra 0,174 no YOLO26s. No último, 5x e 10x arredondam
para 0,193. A relação entre volume e desempenho varia conforme a arquitetura.

Na validação própria, a diferença entre `manual-full` e a melhor condição
sintética é de 0,198 no YOLOv8s, 0,172 no RT-DETR-L e 0,161 no YOLO26s.
A proximidade dessa avaliação com o treino de `manual-full` e seu uso na
seleção de checkpoint favorecem a referência manual.

![F1 por volume sintético](figures/results/synthetic-volume-vs-f1.svg)

As curvas de [precision](figures/results/synthetic-volume-vs-precision.svg) e
[recall](figures/results/synthetic-volume-vs-recall.svg) detalham a composição
do F1. No YOLO26s externo, `manual-full` tem recall de 0,439, contra 0,412 em
`synthetic-5x`. Também apresenta menor MAE de contagem que todas as condições
sintéticas, nas três arquiteturas.

`controlled` tem mAP@.50:.95 entre 0,002 e 0,005 na coleta externa e entre
0,000 e 0,002 na validação própria. Esse controle usa fotos de frutas isoladas
e fundos negativos. A comparação com as cenas sintéticas envolve mudanças de
escala, composição, densidade e quantidade de caixas em conjunto.

## Diagnósticos

### Sobreposição das caixas

![AP por limiar de IoU](figures/results/ap-por-iou.svg)

Nos volumes de maior mAP externo, a vantagem relativa sobre `manual-full`
é maior em IoU 0,75 que em IoU 0,50. O resultado indica melhor concordância
das caixas previstas com o gabarito externo no limiar mais exigente.

As caixas sintéticas derivam da máscara visível final. Esse mecanismo fornece
uma hipótese para a diferença de localização, cuja causa permanece aberta:
a comparação altera tanto as imagens quanto a origem dos rótulos.

### Contagem no YOLO26s

![Contagem prevista contra contagem real](figures/results/contagem-prevista-x-real.svg)

O gráfico reúne `manual-full` e as cinco condições sintéticas do YOLO26s.
Cada ponto representa uma imagem avaliada por uma semente, com as duas sementes
no mesmo painel. A diagonal indica a contagem exata.

A reta ajusta `contagem prevista = inclinação × contagem real`, passando pela
origem. As inclinações vão de 0,23 em `synthetic-2x` a 0,59 em `manual-full`,
com R² de 0,34 a 0,77. Esses valores descrevem a subcontagem nesse conjunto.
Uma calibração para uso exige estimar e avaliar a correção em dados separados.

Os limiares de confiança vêm do melhor F1 da validação de origem de cada
execução. Portanto, o diagnóstico reúne o efeito do modelo e do limiar adotado.
O YOLOv8s em `controlled` apresenta sobrecontagem, com MAE de cerca de 500
frutas por imagem. A direção do erro depende da condição e do detector.

### Progressão da validação por época

![Convergência por época](figures/results/convergencia-por-epoca.svg)

O gráfico mostra mAP@.50 da validação de cada execução, dividido pelo maior
valor observado nas 50 épocas. Reúne `manual-full` e as condições sintéticas.
As medianas da primeira época que alcança 95% desse pico são 22 em
`manual-full`, 19,5 em 1x, 18 em 2x, 17 em 3x, 11 em 5x e 9 em 10x.

Os volumes maiores alcançam esse patamar em menos épocas, cada uma com mais
passos de otimização. A normalização descreve a progressão dentro do orçamento
observado. A duração necessária para obter o melhor desempenho permanece uma
questão distinta.

## Exemplos

Quatro fotos da coleta própria e quatro cenas compostas, selecionadas perto da
densidade mediana de cada coleção, com suas caixas desenhadas:

![Pomar real e cenas compostas](figures/results/sheets/real-x-sintetico.jpg)

Oito cenas sintéticas selecionadas por quantis de quantidade de caixas,
entre uma e 37 frutas. A seleção e os hashes estão no
[registro dos exemplos](figures/results/synthetic-examples/provenance.json).

![Cenas sintéticas](figures/results/sheets/cenas-sinteticas.jpg)

Os exemplos externos usam YOLO26s, semente 41, entrada de 960 pixels e
confiança 0,25. Na cena mediana, o gabarito tem oito frutas. `manual-full`
retorna oito detecções, e as condições sintéticas retornam de quatro a sete.
Essas contagens descrevem as previsões exibidas em cada painel.

![Detecções na cena externa mediana](figures/results/sheets/externo-cena-mediana.jpg)

A [cena densa](figures/results/sheets/externo-cena-densa.jpg) contém 41 frutas.
Os registros da [cena mediana](figures/results/examples/cena-mediana/provenance.json)
e da [cena densa](figures/results/examples/provenance.json) identificam os
checkpoints e os limiares.

### Maior diferença de acertos na validação própria

A seleção compara YOLO26s, semente 41, nas 26 imagens de `manual_full_val`.
Ordena a diferença entre os acertos de `manual-full` e da melhor condição
sintética por imagem, com desempate pelo nome do arquivo.

Em `img_2056`, há 23 frutas no gabarito. Com confiança 0,25 e correspondência
um a um a IoU 0,50, `manual-full` acerta 20 e a melhor sintética acerta 13.
É um dos casos de maior diferença nessa comparação. O
[registro da seleção](figures/results/examples/manual-full-val/yolo26s/gap-audit.json)
contém os acertos, falsos positivos e falsos negativos de todas as imagens.

![Detecções na validação própria](figures/results/sheets/manual-full-val-yolo26s.jpg)

Os recortes ampliados permitem conferir a oclusão, a iluminação e os limites
das caixas. Aparecem da maior para a menor. O número após o ponto indica o
lado maior da caixa dividido pelo menor lado da imagem.

![Frutas do gabarito ampliadas](figures/results/sheets/manual-full-val-zoom.jpg)

Os mapas abaixo mostram a distribuição espacial das anotações:

![Mapas de anotações](figures/results/sheets/mapas-de-anotacoes.jpg)

## Alcance das conclusões

O desvio-padrão amostral de mAP@.50:.95 entre as duas sementes na coleta
externa é:

| Condição | YOLOv8s | RT-DETR-L | YOLO26s |
|---|---:|---:|---:|
| manual-full | 0.0196 | 0.0356 | 0.0303 |
| synthetic-1x | 0.0199 | 0.0193 | 0.0087 |
| synthetic-2x | 0.0346 | 0.0219 | 0.0217 |
| synthetic-3x | 0.0100 | 0.0129 | 0.0051 |
| synthetic-5x | 0.0012 | 0.0040 | 0.0055 |
| synthetic-10x | 0.0089 | 0.0145 | 0.0047 |

As médias e os desvios descrevem estas duas repetições. Os volumes compartilham
um único pool, e a escolha do melhor volume usa os próprios resultados
externos. Uma avaliação independente requer mais sementes, novas gerações e
uma coleta reservada para essa finalidade.

O aumento de volume altera simultaneamente imagens, caixas, tamanho da
validação e passos de otimização. A validação sintética usa ativos conhecidos
enquanto `manual_full_val` reutiliza a validação de `manual-full`. A coleta
externa tem anotação semiautomática e reúne recortes correlacionados de uma
mesma foto. Esses fatores restringem a interpretação das diferenças observadas.

Nesta grade, o treino com cenas compostas e caixas geradas automaticamente
atinge as maiores médias externas de mAP.
`manual-full` conserva vantagem na validação própria e no erro de contagem.
A pergunta sobre reduzir o trabalho de rotulagem com desempenho aceitável
continua aberta, pois exige medir o esforço humano e definir a margem de perda
aceitável para a aplicação.

## Auditar as anotações

Para conferir a estrutura dos rótulos, as contagens e os hashes, execute:

```bash
.venv/bin/python scripts/validate_data.py --stage real
```

Para revisar visualmente a base e registrar os vereditos, execute:

```bash
.venv/bin/python scripts/audit_dataset.py \
	--images data/real_source/images \
	--labels data/real_source/labels \
	--output artifacts/auditoria/manual_full.jsonl --port 8772
```

Abra [127.0.0.1:8772](http://127.0.0.1:8772). As teclas de 1 a 6 marcam `ok`,
`faltando`, `caixa-frouxa`, `duplicada`, `oclusao-extrema` e `nao-e-fruta`.
Enter salva o veredito e avança.

Para começar pelas imagens em que o detector e o gabarito mais discordam,
acrescente `--checkpoint /caminho/best.pt`. A discordância orienta a fila de
inspeção; a decisão cabe ao anotador.

Para contar as marcas da revisão, execute:

```bash
.venv/bin/python scripts/audit_dataset.py \
	--output artifacts/auditoria/manual_full.jsonl --resumo
```

## Reproduzir os resultados

Prepare os dados conforme o [README](../README.md#preparar-e-executar). Para
treinar e selecionar os checkpoints, execute:

```bash
.venv/bin/python scripts/train_grid.py --config configs/confirmatory.yaml --device 0
.venv/bin/python scripts/select_models.py --config configs/confirmatory.yaml
```

Para baixar a coleta externa, aplicar a curadoria e importar o conjunto de
avaliação, execute:

```bash
.venv/bin/python scripts/download_external.py oranges_field
.venv/bin/python -m zipfile -e data/archives/external/oranges-in-the-field.zip artifacts/oranges_field_source
.venv/bin/python scripts/curate_oranges_field.py \
	--source artifacts/oranges_field_source \
	--output artifacts/oranges_field_curated --teto 0.25 --cap 250 --seed 42
.venv/bin/python scripts/import_external_test.py oranges_field \
	--source artifacts/oranges_field_curated
```

Para importar as 26 imagens da validação própria como conjunto de avaliação,
prepare uma pasta com as imagens e seus rótulos:

```bash
mkdir -p artifacts/manual_full_val_source/images artifacts/manual_full_val_source/labels
cp data/real_yolo_confirmatory/images/val/* artifacts/manual_full_val_source/images/
cp data/real_yolo_confirmatory/labels/val/* artifacts/manual_full_val_source/labels/
.venv/bin/python scripts/import_external_test.py manual_full_val \
	--source artifacts/manual_full_val_source
```

Com ambos os conjuntos importados, execute a avaliação e consolide as tabelas:

```bash
for d in oranges_field manual_full_val; do
	.venv/bin/python scripts/evaluate_test.py --config configs/confirmatory.yaml \
		--external-name "$d" --device 0 --unlock-test
done
.venv/bin/python scripts/plot_confirmatory_results.py
```

O pacote de checkpoints configurado em `pipeline.yaml` pode ser obtido com
`.venv/bin/python scripts/download_weights.py`. Ele fornece os pesos e a seleção
usados na avaliação. Para executar uma arquitetura por vez, passe `--model` a
`train_grid.py` e a `select_models.py`.

As tabelas consolidadas ficam em `artifacts/confirmatory/results_tables.md`.
Os diagnósticos leem `run_metrics.csv`, `counting_by_image.csv` e
`training_history.csv` em `artifacts/confirmatory/analysis_csv/`. Para exportar esses CSVs dos
resultados e históricos de treino e gerar os gráficos, execute:

```bash
.venv/bin/python scripts/generate_report.py --external-name oranges_field
.venv/bin/python scripts/plot_grid_diagnostics.py
```

Para reproduzir a seleção do caso de maior diferença e os exemplos, execute:

```bash
.venv/bin/python scripts/find_detection_gap.py
.venv/bin/python scripts/render_detection_examples.py --dataset manual_full_val --model yolo26s --seed 41 --image-stem img_2056
.venv/bin/python scripts/render_synthetic_examples.py
.venv/bin/python scripts/build_example_sheets.py
```

Para conferir os links e a consistência dos números publicados com os SVGs,
execute `.venv/bin/python scripts/check_documentation.py`. Para conferir também
as sete métricas das tabelas com os JSONs da rodada, acrescente
`--results-dir artifacts/confirmatory`.
