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

*Figura 1 — Preparação das imagens, composição das cenas e formação dos conjuntos sintéticos. Os parâmetros citados nas caixas estão detalhados na seção seguinte.*

## Parâmetros do gerador

Todos ficam em [`configs/synthesis/confirmatory_pool.yaml`](configs/synthesis/confirmatory_pool.yaml)
e são consumidos por `fruit_pipeline/synthesis.py`. A geração é determinística:
mesma configuração e mesma `seed` produzem exatamente o mesmo conjunto, e
qualquer alteração aqui muda o hash da configuração, o que invalida o cache e
força o retreino apenas das condições afetadas.

### Cena

| Parâmetro | O que faz |
|---|---|
| `seed` | Semente única de toda a geração. Cada cena deriva sua própria semente de `seed` + hash da configuração + índice da cena. |
| `images.total` | Tamanho do pool antes do split. As frações `1x`–`10x` são prefixos aninhados desse pool. |
| `canvas` | Resolução `[altura, largura]` de cada cena. O fundo é redimensionado para esse tamanho. |

### Quantidade e tamanho dos objetos (`objects`)

| Parâmetro | O que faz |
|---|---|
| `min` / `max` | Faixa esparsa de frutas por cena, sorteada uniformemente. |
| `dense.probability` | Probabilidade de a cena usar a faixa densa em vez da esparsa. Produz uma distribuição bimodal que cobre tanto foto de perto quanto pomar fotografado de longe. |
| `dense.min` / `dense.max` | Faixa densa de frutas por cena. O teto é limitado pela VRAM: acima de ~200 instâncias por imagem o RT-DETR não treina em 8 GB. |
| `min_scale` / `max_scale` | Tamanho aparente da fruta como fração linear (`√(l·a)`) do canvas. Calibrado contra a distribuição real de caixas dos dois testes externos. |
| `scale_mode` | `canvas` escala em relação à imagem; `cutout` escala em relação ao próprio recorte. |
| `rotation_degrees` | Rotação máxima em graus, sorteada por instância em `[-x, +x]`. Também randomiza a direção do brilho especular, que é idêntica nos recortes-fonte. |
| `depth_scale.enabled` | Liga a modulação do tamanho pela profundidade do ponto de inserção. |
| `depth_scale.near_scale` / `far_scale` | Multiplicadores do tamanho para fruta no primeiro plano e no fundo. Interpolados linearmente pela proximidade estimada. |

### Colocação (`placement`)

A profundidade estimada pelo DepthPro define um valor `z` para cada fruta; tudo
no fundo que estiver mais próximo que esse `z` passa a ocluí-la.

| Parâmetro | O que faz |
|---|---|
| `z_method` | Como derivar o `z` da fruta a partir da profundidade sob sua silhueta. `center_patch` usa a mediana de um retalho ao redor do ponto de ancoragem; `quantile` e `mean_plus_std` são alternativas. |
| `z_patch_fraction` | Tamanho do retalho central, como fração do menor lado da fruta. Mediana pequena é robusta a ruído de um pixel sem perder o significado de eixo Z. |
| `z_offset` | Deslocamento fixo do `z`. Negativo empurra a fruta para trás, aumentando a oclusão. |
| `z_offset_jitter` | Faixa de sorteio do offset por tentativa. Sem ele quase nenhuma fruta sai totalmente visível nem totalmente oculta, porque a variação viria só da geometria local. |
| `min_depth` | Profundidade mínima aceita no ponto de inserção. Rejeita colocações no céu ou em regiões sem estrutura. |
| `min_visibility` | Fração mínima da fruta que precisa sobrar visível após a oclusão. Abaixo disso a inserção é descartada. |
| `max_attempts_per_object` | Tentativas de posição por fruta antes de desistir dela. |
| `exclude_bottom_fraction` | Faixa inferior da imagem onde não se coloca fruta, evitando fruta flutuando sobre o chão em primeiro plano. |

### Aparência (`appearance`)

Aplicada nesta ordem: maturação → casting ambiental → exposição por instância.

| Parâmetro | O que faz |
|---|---|
| `ripeness.enabled` | Liga o deslocamento de matiz que simula fruta em maturação. Os 127 recortes-fonte são todos de fruta madura. |
| `ripeness.fraction_affected` | Fração das instâncias que recebe o deslocamento. |
| `ripeness.green_hue_degrees` | Matiz alvo em graus. Um verde-amarelado, deliberadamente distinto do verde da folhagem para não colidir cromaticamente com ela. |
| `ripeness.strength_range` | Intensidade do deslocamento por instância, de 0 (sem efeito) a 1 (matiz alvo puro). |
| `ripeness.saturation_scale` | Multiplicador da saturação. Mantém a fruta verde menos saturada que a folha real, preservando a distinção. |
| `ripeness.gloss_reduction` | Achata o brilho especular acima do percentil 75 de luminância do recorte. Fruta "de vez" é mais fosca que a madura. |
| `hsv_cast.enabled` | Liga a integração da fruta à luz da cena, puxando matiz e saturação para um alvo derivado do fundo local. |
| `hsv_cast.use_hardlight_target` | Usa o hard-light da fruta contra a cor média do fundo como alvo por pixel, em vez de uma cor única e plana. |
| `hsv_cast.hue_power` / `saturation_power` / `value_power` | Quanto cada canal HSV adota o alvo ambiental, de 0 a 1. |
| `hsv_cast.value_power_jitter` | Sorteio do `value_power` por instância, alargando a variação de luz e sombra entre frutas. |
| `hsv_cast.min_value_ratio` | Piso de luminância relativo ao valor original, impedindo que a fruta colapse num borrão indistinguível do fundo. |
| `hsv_cast.bright_flatten_strength` | Perto de regiões estouradas de luz, aumenta a adoção do alvo, achatando o relevo como acontece na superexposição real. |
| `exposure_jitter.enabled` | Liga o fator de exposição por instância, aplicado depois do casting e independente do fundo. |
| `exposure_jitter.probability` | Fração das instâncias afetadas. |
| `exposure_jitter.range` | Faixa do multiplicador de luminância. Abaixo de 1 produz fruta em sombra profunda; acima, fruta em sol direto. É o que reproduz o espalhamento de contraste observado nas fotos reais, que o `hsv_cast` sozinho não consegue por só saber reduzir contraste. |
| `exposure_jitter.saturation_pull` | Dessaturação proporcional ao afastamento de 1. Sombra profunda e sol direto lavam a cor, por motivos opostos. |
| `hardlight_power` | Intensidade do hard-light no modo legado, usado quando `hsv_cast` está desligado. |

### Oclusão e sombras (`occlusion`)

| Parâmetro | O que faz |
|---|---|
| `edge_blur` | Suaviza a máscara de visibilidade derivada da profundidade, evitando bordas de oclusão em degrau. |
| `depth_smooth_radius` | Suaviza o mapa de profundidade antes de compará-lo ao `z`, reduzindo ruído do DepthPro. |
| `mask_threshold` | Limiar que converte a máscara suavizada em oclusão efetiva. |
| `edge_feather_radius` | Segundo desfoque, mais curto, no contorno do recorte, para não sobrar franja semitransparente. |
| `contact_shadow.enabled` | Liga a penumbra curta na faixa vizinha à região ocluída. Uma folha à frente não só recorta a fruta, também bloqueia luz ao redor. |
| `contact_shadow.strength` | Intensidade do escurecimento. |
| `contact_shadow.radius_fraction` | Alcance da penumbra como fração do menor lado da fruta. |
| `cast_shadow.enabled` | Liga a sombra projetada sobre a fruta por estruturas fora do recorte. |
| `cast_shadow.probability` | Fração das instâncias que recebe sombra projetada. |
| `cast_shadow.strength` | Intensidade do escurecimento. |
| `cast_shadow.min_coverage` / `max_coverage` | Fração da fruta coberta pela sombra, sorteada na faixa. |
| `cast_shadow.offset_fraction` | Deslocamento da sombra em relação ao centro da fruta. |
| `cast_shadow.light_angle_degrees` | Direção da luz em graus, que define de que lado a sombra cai. |
| `cast_shadow.light_angle_jitter_degrees` | Variação do ângulo por instância. |
| `cast_shadow.blur_radius` | Suavidade da borda da sombra. |

### Anotação e saída

| Parâmetro | O que faz |
|---|---|
| `annotation.mode` | `visible` anota apenas a parte visível da fruta; `amodal`, a extensão completa incluindo o que está oculto; `rect`, o retângulo do recorte. |
| `annotation.min_box_pixels` | Área mínima em pixels para a caixa ser registrada. Descarta fruta praticamente invisível. |
| `output.jpeg_quality` | Qualidade JPEG da cena final. |
| `scene_grading.enabled` | Liga a correção aplicada à cena inteira, fruta e fundo juntos, depois da composição. |
| `scene_grading.contrast` / `saturation` / `brightness` | Multiplicadores globais. Aplicados à imagem toda, unificam fruta e fundo sob a mesma resposta tonal. |
| `scene_grading.sharpen_radius` / `sharpen_percent` / `sharpen_threshold` | Máscara de nitidez final, equalizando a nitidez do recorte com a do fundo fotográfico. |


## Experimento

O experimento compara sete condições de treinamento:

| Condição | Conjuntos |
|---|---|
| <code>manual&#8209;full</code> | fotografias de campo<br><img src="docs/figures/condicoes/condicao-manual-full.svg" width="1092" alt="104 imagens de treino, 26 de validação e 119 do teste externo CitDet"> |
| <code>controlled</code> | frutas isoladas + fundos negativos<br><img src="docs/figures/condicoes/condicao-controlled.svg" width="1092" alt="284 imagens de treino, 71 de validação e 119 do teste externo CitDet"> |
| <code>synthetic&#8209;1x</code> | cenas sintéticas<br><img src="docs/figures/condicoes/condicao-synthetic-1x.svg" width="1092" alt="104 imagens de treino, 26 de validação e 119 do teste externo CitDet"> |
| <code>synthetic&#8209;2x</code> | contém synthetic-1x<br><img src="docs/figures/condicoes/condicao-synthetic-2x.svg" width="1092" alt="208 imagens de treino, 52 de validação e 119 do teste externo CitDet"> |
| <code>synthetic&#8209;3x</code> | contém synthetic-2x<br><img src="docs/figures/condicoes/condicao-synthetic-3x.svg" width="1092" alt="312 imagens de treino, 78 de validação e 119 do teste externo CitDet"> |
| <code>synthetic&#8209;5x</code> | condição sintética principal<br><img src="docs/figures/condicoes/condicao-synthetic-5x.svg" width="1092" alt="520 imagens de treino, 130 de validação e 119 do teste externo CitDet"> |
| <code>synthetic&#8209;10x</code> | análise de saturação<br><img src="docs/figures/condicoes/condicao-synthetic-10x.svg" width="1092" alt="1.040 imagens de treino, 260 de validação e 119 do teste externo CitDet"> |

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
- Os 42 treinamentos confirmatórios foram executados e avaliados contra dois
  testes externos (CitDet e o split de validação de `manual-full`). Resultados,
  gráficos e mapas de calor em [`docs/RESULTS.md`](docs/RESULTS.md).

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
