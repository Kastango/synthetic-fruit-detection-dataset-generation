# Detecção de poncãs com dados sintéticos

Pipeline reprodutível para investigar se imagens sintéticas podem substituir
imagens manualmente anotadas no treinamento de detectores de poncãs.

## Experimento

O experimento compara sete condições de treinamento:

| Condição | Imagens de treino | Conteúdo |
|---|---:|---|
| `manual-full` | 104 | fotografias de campo anotadas manualmente |
| `controlled` | 284 | frutas fotografadas em ambiente controlado e fundos negativos |
| `synthetic-1x` | 104 | cenas sintéticas |
| `synthetic-2x` | 208 | cenas sintéticas |
| `synthetic-3x` | 312 | cenas sintéticas |
| `synthetic-5x` | 520 | condição sintética principal |
| `synthetic-10x` | 1.040 | análise de saturação |

Os conjuntos sintéticos são aninhados: `2x` contém `1x`, `3x` contém `2x` e
assim por diante. O pool completo possui 1.040 imagens de treino e 260 de
validação.

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
contagem. Resultados sintéticos superiores a `manual-full` são destacados no
relatório final.

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

`datanotation.zip` preserva a resolução original das câmeras e não contém
duplicatas ou augmentation. A exportação `'Ponca 3 v2.zip` não é utilizada,
pois mistura imagens reais processadas, imagens sintéticas e augmentation do
Roboflow.

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

Se um arquivo já estiver disponível no servidor, informe-o diretamente:

```bash
./run_pipeline.sh all \
  --real-source /datasets/datanotation.zip \
  --external-source /datasets/UTA_CSE_Dataset.zip \
  --device 0 --accept-data-terms --unlock-test
```

O número de processos auxiliares é escolhido automaticamente a partir dos CPUs
disponíveis.

## Etapas da pipeline

1. Baixar e validar as fontes.
2. Preparar os conjuntos manual, controlado e sintéticos.
3. Treinar os 42 modelos e registrar tempo, configuração e métricas.
4. Selecionar os checkpoints pela validação de origem.
5. Preparar o teste externo e avaliar todos os modelos.
6. Gerar o relatório consolidado.

A execução pode ser retomada. Dados já preparados, treinamentos concluídos e
checkpoints intermediários são reutilizados.

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
