# Detecção de poncãs com dados sintéticos

Pipeline reprodutível para gerar conjuntos sintéticos e compará-los com dados
anotados manualmente na detecção de poncãs.

## Problema e pergunta de pesquisa

Treinar um detector de frutas exige imagens da época de frutificação e caixas
desenhadas à mão. A base real usada aqui tem 130 imagens e 2.093 caixas. Cobrir
diferentes condições de iluminação e oclusão exige novas coletas e mais
rotulagem.

> Imagens sintéticas anotadas automaticamente podem reduzir o esforço de
> rotulagem e manter desempenho próximo ao obtido com imagens reais?

A pipeline combina fotos de árvores sem frutas, feitas fora do período
produtivo, com frutas fotografadas sobre fundo uniforme. O DepthPro estima a
profundidade, e o compositor usa o mapa para posicionar as frutas, simular
oclusões, ajustar sua aparência e gerar as caixas. O processo usa imagens RGB
comuns, sem sensores de profundidade ou modelagem 3D, e gera os rótulos junto
com cada cena.

Os dados reais são a referência. Os conjuntos sintéticos variam de `1x` a
`10x` para medir como o desempenho muda com o volume de dados.

## Fluxo de geração

O fluxograma resume a preparação dos ativos, a composição das cenas, os arquivos
salvos em JPG, TXT e JSON, o split do pool gerado e a materialização de
`synthetic-1x` a `synthetic-10x`.

![Fluxograma da geração dos conjuntos synthetic-1x a synthetic-10x](docs/figures/fluxograma-geracao-conjuntos-sinteticos.svg)

*Figura 1 — Preparação das imagens, composição das cenas e formação dos conjuntos sintéticos.*

## Experimento

O experimento compara sete condições de treinamento:

| Condição | Conjuntos | Conteúdo |
|---|---|---|
| `manual-full` | <img src="docs/figures/condicoes/condicao-manual-full.svg" width="524" alt="104 imagens de treino, 26 de validação e 119 do teste externo CitDet"> | fotografias de campo anotadas manualmente |
| `controlled` | <img src="docs/figures/condicoes/condicao-controlled.svg" width="524" alt="284 imagens de treino, 71 de validação e 119 do teste externo CitDet"> | frutas fotografadas em ambiente controlado e fundos negativos |
| `synthetic-1x` | <img src="docs/figures/condicoes/condicao-synthetic-1x.svg" width="524" alt="104 imagens de treino, 26 de validação e 119 do teste externo CitDet"> | cenas sintéticas |
| `synthetic-2x` | <img src="docs/figures/condicoes/condicao-synthetic-2x.svg" width="524" alt="208 imagens de treino, 52 de validação e 119 do teste externo CitDet"> | cenas sintéticas |
| `synthetic-3x` | <img src="docs/figures/condicoes/condicao-synthetic-3x.svg" width="524" alt="312 imagens de treino, 78 de validação e 119 do teste externo CitDet"> | cenas sintéticas |
| `synthetic-5x` | <img src="docs/figures/condicoes/condicao-synthetic-5x.svg" width="524" alt="520 imagens de treino, 130 de validação e 119 do teste externo CitDet"> | condição sintética principal |
| `synthetic-10x` | <img src="docs/figures/condicoes/condicao-synthetic-10x.svg" width="524" alt="1.040 imagens de treino, 260 de validação e 119 do teste externo CitDet"> | análise de saturação |

Todos os fundos, mapas de profundidade e recortes ficam disponíveis durante a
composição. O gerador cria um pool único de 1.300 cenas e só depois aplica o
split determinístico 80/20: 1.040 imagens de treino e 260 de validação. Os
conjuntos são aninhados nas duas partições: `2x` contém o treino e a validação
de `1x`, `3x` contém os de `2x` e assim por diante. Cada multiplicador usa apenas
o prefixo de validação correspondente ao seu tamanho.

Cada condição é treinada com três detectores:

| Família | Checkpoint |
|---|---|
| YOLO26 | `yolo26s.pt` |
| YOLOv8 | `yolov8s.pt` |
| RT-DETR | `rtdetr-l.pt` |

Cada treinamento executa no máximo 50 épocas e pode parar antes caso a validação
não melhore por 30 épocas. Todos usam entrada `960`, a mesma política de
augmentation e as sementes 41 e 42. A matriz completa contém:

```text
7 condições × 3 detectores × 2 sementes = 42 treinamentos
```

## Avaliação

O teste externo usa as 119 imagens e 10.082 caixas do split oficial de teste do
[CitDet](https://mavmatrix.uta.edu/cse_datasets/1/). O split de treino do CitDet
não é utilizado.

Os checkpoints são selecionados pela validação correspondente a cada condição.
O teste externo só é preparado depois que essa seleção é congelada em
`model_selection.json`.

A métrica principal é mAP@0.5:0.95. O relatório também inclui precisão,
revocação, F1, mAP@0.5, mAP@0.75, AP por IoU, tempo de inferência e erros de
contagem, curvas de treinamento e um mapa de calor das anotações de cada
conjunto. Os dados completos da análise também são exportados em CSV e reunidos
em `analysis_csv.zip`. Resultados sintéticos superiores a `manual-full` são
destacados no relatório final.

## Estado atual

- Pipeline confirmatória implementada e validada por testes automatizados.
- Dry-run confirmado com 42 treinamentos.
- Treinamento confirmatório e avaliação final ainda não executados.

## Dados

| Fonte | Uso | Conteúdo |
|---|---|---|
| `datanotation.zip` | treino e validação manual | 130 imagens, 2.093 caixas YOLO, sendo 82 fotos do iPhone 13 mini e 48 do Pixel 6a |
| ativos sintéticos | condição controlada e geração de cenas | 127 fotos de frutas, 228 fundos e seus mapas de profundidade |
| `UTA_CSE_Dataset.zip` | teste externo | split oficial do CitDet com 119 imagens e 10.082 caixas |

Os arquivos necessários são baixados automaticamente e validados por tamanho e
SHA-256. Os endereços e hashes estão em
[`configs/pipeline.yaml`](configs/pipeline.yaml). A auditoria completa está em
[`docs/DATASETS.md`](docs/DATASETS.md).

## Execução

Requer Python 3.11 ou 3.12 e uma GPU compatível com CUDA. O script abaixo cria o
ambiente virtual, instala as dependências e executa a pipeline.

Confira a configuração sem baixar dados ou iniciar treinos:

```bash
./run_pipeline.sh all --dry-run --device 0 --accept-data-terms
```

Execute o experimento completo:

```bash
./run_pipeline.sh all \
  --device 0 \
  --accept-data-terms \
  --unlock-test
```

`--unlock-test` autoriza a avaliação externa depois que os checkpoints forem
selecionados. Sem essa opção, a pipeline termina após a seleção.

O número de processos auxiliares é escolhido automaticamente a partir dos CPUs
disponíveis.

## Etapas da pipeline

1. Baixar e validar as fontes.
2. Gerar os recortes e mapas DepthPro e preparar os conjuntos.
3. Treinar os 42 modelos e registrar tempo, configuração e métricas.
4. Selecionar os checkpoints pela validação de origem.
5. Preparar o teste externo e avaliar todos os modelos.
6. Gerar o relatório consolidado.

A execução pode ser retomada repetindo o mesmo comando. Downloads parciais
continuam de onde pararam; arquivos e datasets completos são validados e
reutilizados; treinamentos interrompidos retomam do último checkpoint.

A pipeline informa a etapa atual a cada cinco minutos. O ETA usa o tempo da
mesma etapa em execuções anteriores no servidor e fica disponível depois do
primeiro registro completo.

## Outros conjuntos de teste

Novos testes podem ser registrados em `external_datasets`, dentro de
[`configs/pipeline.yaml`](configs/pipeline.yaml), e avaliados sem retreinar os
modelos:

```bash
./run_pipeline.sh prepare-test \
  --external-name oranges_mendeley \
  --external-source /datasets/oranges-in-the-field.zip

./run_pipeline.sh test --device 0 --unlock-test \
  --external-name oranges_mendeley

./run_pipeline.sh report --external-name oranges_mendeley
```

## Principais arquivos

| Arquivo | Finalidade |
|---|---|
| `configs/pipeline.yaml` | fontes, caminhos e validações dos dados |
| `configs/confirmatory.yaml` | condições, modelos e parâmetros de treino |
| `configs/synthesis/confirmatory_pool.yaml` | configuração do gerador sintético |
| `scripts/reproduce.py` | orquestração das etapas |

Os resultados são salvos em `artifacts/confirmatory/`. O relatório final fica
em `artifacts/confirmatory/RESULTS_citdet.md`.

## Verificação local

```bash
.venv/bin/python -m pytest -q
uvx ruff check .
```
