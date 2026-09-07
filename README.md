# Detecção de poncãs com dados sintéticos

Pipeline para gerar imagens sintéticas com caixas automáticas e comparar
seu uso no treinamento de detectores de poncãs. A avaliação no CitDet mede
transferência para detecção de cítricos, com todas as categorias reunidas em
uma classe.

Os 42 treinamentos previstos foram concluídos. As tabelas e os exemplos estão
em [`docs/RESULTS.md`](docs/RESULTS.md). Os resultados são exploratórios: o
gerador foi ajustado com estatísticas dos conjuntos de avaliação. O nome
`confirmatory` identifica os arquivos da pipeline, não garante independência
experimental.

## Problema e pergunta de pesquisa

O treinamento supervisionado de um detector de frutas pode depender de
imagens da época de frutificação e caixas desenhadas à mão. A base real usada
aqui tem 130 imagens e 2.093 caixas. Cobrir
diferentes condições de iluminação e oclusão exige novas coletas e mais
rotulagem.

> Imagens sintéticas anotadas automaticamente podem reduzir o esforço de
> rotulagem e manter desempenho próximo ao obtido com imagens reais?

A pipeline combina fotos de árvores sem frutas, feitas fora do período
produtivo, com frutas fotografadas sobre fundo uniforme. O
[DepthPro](https://github.com/apple/ml-depth-pro) estima a
profundidade, e o compositor usa o mapa para posicionar as frutas, simular
oclusões, ajustar sua aparência e gerar as caixas. O processo usa imagens RGB
comuns, sem sensores de profundidade ou modelagem 3D, e gera os rótulos junto
com cada cena. Isso dispensa desenhar cada caixa sintética à mão; ainda exige
coleta, pré-processamento e auditoria das imagens. O esforço total de trabalho
humano não foi medido.

Os dados reais são a referência. Os conjuntos sintéticos variam de `1x` a
`10x` para comparar volumes sob o protocolo descrito abaixo. A quantidade de
caixas, o tamanho da validação e os passos de otimização também variam.

## Fluxo de geração

O fluxograma resume a preparação dos ativos, a composição das cenas, os arquivos
salvos em JPG, TXT e JSON, o split do pool gerado e a materialização de
`synthetic-1x` a `synthetic-10x`.

![Fluxograma da geração dos conjuntos synthetic-1x a synthetic-10x](docs/figures/fluxograma-geracao-conjuntos-sinteticos.svg)

*Figura 1. Preparação dos ativos, composição e formação dos conjuntos sintéticos.
As faixas do laço mostram a mesma fruta, fundo e posição em uma inserção
ilustrativa. O split separa cenas, mas permite reutilizar os mesmos ativos em
treino e validação. [Abrir o SVG em tamanho completo](docs/figures/fluxograma-geracao-conjuntos-sinteticos.svg).*

Os scripts, fontes e miniaturas estão versionados. Veja
[como editar e regenerar os fluxogramas](docs/DIAGRAMS.md).

## Parâmetros do gerador

<details>
<summary>Consultar os parâmetros e seu significado no código</summary>

Todos ficam em [`configs/synthesis/confirmatory_pool.yaml`](configs/synthesis/confirmatory_pool.yaml)
e são consumidos por [`fruit_pipeline/synthesis.py`](fruit_pipeline/synthesis.py).
A geração usa sementes por cena e registra hashes da configuração e do catálogo
de ativos. A reprodução depende também das mesmas fontes, versão do código e
bibliotecas. Mudanças no gerador exigem regenerar os dados e conferir os
manifestos antes de reutilizar treinamentos; o hash da configuração, sozinho,
não detecta uma alteração no código.

### Cena

| Parâmetro | O que faz |
|---|---|
| `seed` | Semente única de toda a geração. Cada cena deriva sua própria semente de `seed` + hash da configuração + índice da cena. |
| `images.total` | Tamanho do pool antes do split. As frações `1x`–`10x` são prefixos aninhados desse pool. |
| `canvas` | Resolução `[largura, altura]`, como em Pillow. O valor atual `[720, 960]` produz imagens em retrato. |

### Quantidade e tamanho dos objetos (`objects`)

| Parâmetro | O que faz |
|---|---|
| `min` / `max` | Faixa esparsa de frutas solicitadas, sorteada uniformemente entre 1 e 30. Inserções podem ser rejeitadas. |
| `dense.probability` | Probabilidade de a cena sortear da faixa densa em vez da esparsa, atualmente 6%. A distribuição final depende das rejeições e oclusões. |
| `dense.min` / `dense.max` | Faixa densa de frutas solicitadas, de 60 a 110. Junto com a faixa esparsa (`min`/`max`), produz uma mistura de dois modos: a maioria das cenas fica perto das 14,5 caixas por imagem medianas do `manual-full` e uma minoria perto das 78 do CitDet. Em 104 imagens isso dá cerca de 2.100 caixas, próximo das 2.093 do conjunto real. |
| `min_scale` / `max_scale` | Fração do menor lado do canvas que define o maior lado do recorte antes da rotação e do ajuste por profundidade. Faixa atual 0,01–0,065, calibrada com estatísticas de caixas do CitDet e da base manual. |
| `rotation_degrees` | Rotação no plano da imagem, sorteada em `[-x, +x]` por instância. Gira também o brilho registrado na foto; não recalcula a iluminação em 3D. |
| `depth_scale.near_scale` / `far_scale` | Multiplicadores do tamanho para fruta no primeiro plano e no fundo. Interpolados linearmente pela proximidade estimada. |

Quando a quantidade solicitada supera os 127 recortes disponíveis, o gerador
usa o catálogo completo e sorteia os recortes adicionais com repetição. Uma
foto-fonte pode, portanto, originar várias instâncias na mesma cena.

### Colocação (`placement`)

O pré-processamento converte a distância estimada pelo DepthPro em proximidade
normalizada por imagem, de 0 a 255, com valores maiores indicando regiões mais
próximas. O compositor deriva um `z` local para a fruta; pixels do fundo com
proximidade maior que esse valor ocluem sua silhueta. `z` e os limiares abaixo
não representam metros.

| Parâmetro | O que faz |
|---|---|
| `z_patch_fraction` | Lado do patch central, como fração do menor lado da fruta. Sua mediana define uma referência local de proximidade. |
| `z_offset` | Deslocamento fixo do `z`. Negativo empurra a fruta para trás, aumentando a oclusão. |
| `z_offset_jitter` | Amplitude da perturbação uniforme somada ao offset por tentativa, em unidades do mapa de 8 bits. |
| `min_depth` | Proximidade mínima aceita no ponto de inserção, na escala 0–255. Não é uma segmentação semântica do céu. |
| `min_visibility` | Fração mínima de pixels da silhueta visível na inserção. Frutas posteriores podem reduzir essa fração; o limiar de 15% não é reaplicado às caixas finais. |
| `max_attempts_per_object` | Tentativas de posição por fruta antes de desistir dela. |
| `exclude_bottom_fraction` | Fração inferior excluída da amostragem de posições, usada como aproximação para evitar o chão. |

### Aparência (`appearance`)

Aplicada nesta ordem: variação de matiz → HSV cast ambiental → exposição por
instância. São transformações de aparência, sem simulação física da maturação
ou da iluminação.

| Parâmetro | O que faz |
|---|---|
| `ripeness.enabled` | Liga o deslocamento de matiz que simula fruta em maturação. Os 127 recortes-fonte são todos de fruta madura. |
| `ripeness.fraction_affected` | Fração das instâncias que recebe o deslocamento. |
| `ripeness.green_hue_degrees` | Matiz alvo em graus. Um verde-amarelado, deliberadamente distinto do verde da folhagem para não colidir cromaticamente com ela. |
| `ripeness.strength_range` | Intensidade do deslocamento por instância, de 0 (sem efeito) a 1 (matiz alvo puro). |
| `ripeness.saturation_scale` | Multiplicador da saturação. Controla a saturação da transformação; não garante separação cromática da folhagem. |
| `ripeness.gloss_reduction` | Reduz valores altos de luminância do recorte. É uma heurística visual, sem calibração de propriedades físicas da casca. |
| `hsv_cast.use_hardlight_target` | Usa o hard-light da fruta contra a cor média do fundo como alvo por pixel, em vez de uma cor única e plana. |
| `hsv_cast.hue_power` / `saturation_power` / `value_power` | Quanto cada canal HSV adota o alvo ambiental, de 0 a 1. |
| `hsv_cast.value_power_jitter` | Sorteio do `value_power` por instância, alargando a variação de luz e sombra entre frutas. |
| `hsv_cast.min_value_ratio` | Piso de luminância relativo ao valor original, impedindo que a fruta colapse num borrão indistinguível do fundo. |
| `hsv_cast.bright_flatten_strength` | Perto de regiões estouradas de luz, aumenta a adoção do alvo, achatando o relevo como acontece na superexposição real. |
| `exposure_jitter.enabled` | Liga o fator de exposição por instância, aplicado depois do casting e independente do fundo. |
| `exposure_jitter.probability` | Fração das instâncias afetadas. |
| `exposure_jitter.range` | Faixa do multiplicador de intensidade, atualmente 0,40–1,90. Escurece ou clareia a fruta independentemente do fundo; não demonstra, por si só, equivalência à distribuição real de iluminação. |
| `exposure_jitter.saturation_pull` | Dessaturação proporcional ao afastamento do fator de exposição em relação a 1. |

### Oclusão e sombras (`occlusion`)

| Parâmetro | O que faz |
|---|---|
| `edge_blur` | Suaviza a máscara de visibilidade derivada da profundidade, evitando bordas de oclusão em degrau. |
| `depth_smooth_radius` | Suaviza o mapa de profundidade antes de compará-lo ao `z`, reduzindo ruído do DepthPro. |
| `mask_threshold` | Limiar que converte a máscara suavizada em oclusão efetiva. |
| `edge_feather_radius` | Segundo desfoque, mais curto, no contorno do recorte, para não sobrar franja semitransparente. |
| `contact_shadow.strength` | Intensidade do escurecimento. |
| `contact_shadow.radius_fraction` | Alcance da penumbra como fração do menor lado da fruta. |
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
| `annotation.min_box_pixels` | Comprimento mínimo de cada lado da caixa, em pixels. Com valor 2, largura e altura precisam ser pelo menos 2 px. |
| `output.jpeg_quality` | Qualidade JPEG da cena final. |
| `output.scene_grading.contrast` / `saturation` / `brightness` | Multiplicadores de contraste, saturação e brilho aplicados somente ao fundo. |
| `output.scene_grading.sharpen_radius` / `sharpen_percent` / `sharpen_threshold` | Máscara de nitidez aplicada ao fundo antes da composição. |


</details>

## Experimento

O experimento compara sete condições de treinamento:

| Condição | Conjuntos |
|---|---|
| <code>manual&#8209;full</code> | fotografias de campo<br><a href="docs/figures/condicoes/condicao-manual-full.svg"><img src="docs/figures/condicoes/condicao-manual-full.svg" width="1092" alt="104 imagens de treino, 26 de validação e 119 do teste externo CitDet"></a> |
| <code>controlled</code> | frutas isoladas + fundos negativos<br><a href="docs/figures/condicoes/condicao-controlled.svg"><img src="docs/figures/condicoes/condicao-controlled.svg" width="1092" alt="284 imagens de treino, 71 de validação e 119 do teste externo CitDet"></a> |
| <code>synthetic&#8209;1x</code> | cenas sintéticas<br><a href="docs/figures/condicoes/condicao-synthetic-1x.svg"><img src="docs/figures/condicoes/condicao-synthetic-1x.svg" width="1092" alt="104 imagens de treino, 26 de validação e 119 do teste externo CitDet"></a> |
| <code>synthetic&#8209;2x</code> | contém synthetic-1x<br><a href="docs/figures/condicoes/condicao-synthetic-2x.svg"><img src="docs/figures/condicoes/condicao-synthetic-2x.svg" width="1092" alt="208 imagens de treino, 52 de validação e 119 do teste externo CitDet"></a> |
| <code>synthetic&#8209;3x</code> | contém synthetic-2x<br><a href="docs/figures/condicoes/condicao-synthetic-3x.svg"><img src="docs/figures/condicoes/condicao-synthetic-3x.svg" width="1092" alt="312 imagens de treino, 78 de validação e 119 do teste externo CitDet"></a> |
| <code>synthetic&#8209;5x</code> | volume intermediário<br><a href="docs/figures/condicoes/condicao-synthetic-5x.svg"><img src="docs/figures/condicoes/condicao-synthetic-5x.svg" width="1092" alt="520 imagens de treino, 130 de validação e 119 do teste externo CitDet"></a> |
| <code>synthetic&#8209;10x</code> | maior volume avaliado<br><a href="docs/figures/condicoes/condicao-synthetic-10x.svg"><img src="docs/figures/condicoes/condicao-synthetic-10x.svg" width="1092" alt="1.040 imagens de treino, 260 de validação e 119 do teste externo CitDet"></a> |

As fotos nas pilhas ilustram os tipos de dados. As miniaturas sintéticas vêm de
um preview do gerador atual e não identificam as cenas dos treinamentos
publicados. Treino e validação usam a mesma escala visual, 3 cartas por 26
imagens, com arredondamento; o teste usa 3 cartas fixas. As setas de ida e volta
representam avaliação entre épocas, sem atualização de pesos pela validação.

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

Cada treinamento executa no máximo 50 épocas, com `patience=30`, entrada
`960` e sementes 41 e 42. Os parâmetros de augmentation são compartilhados na
configuração; sua aplicação depende da implementação de cada detector. YOLOs
usam SGD, `lr0=0.01`, batch 8 e `deterministic=true`; RT-DETR usa AdamW,
`lr0=0.0001`, batch 2 e `deterministic=false`. As comparações de volume devem
ser feitas dentro de cada detector. Todos partem de pesos pré-treinados, não de
inicialização aleatória. A matriz completa contém:

```text
7 condições × 3 detectores × 2 sementes = 42 treinamentos
```

## Avaliação

A avaliação no CitDet usa as 119 imagens e 10.082 caixas do split oficial de teste do
[CitDet](https://mavmatrix.uta.edu/cse_datasets/1/). O split de treino do CitDet
não é utilizado.

Os checkpoints são selecionados pela validação correspondente a cada condição.
O fluxo padrão libera a avaliação externa depois que a seleção é congelada em
`model_selection.json`. Esse bloqueio protege a seleção de checkpoints, mas não
impede que estatísticas do conjunto de avaliação orientem o desenvolvimento
do gerador, como ocorreu nesta versão.

A métrica principal é mAP@0.5:0.95. O relatório também inclui precisão,
revocação, F1, mAP@0.5, mAP@0.75, AP por IoU, tempo de inferência e erros de
contagem, curvas de treinamento e um mapa de calor das anotações de cada
conjunto. Os dados completos da análise também são exportados em CSV e reunidos
em `analysis_csv.zip`. Resultados sintéticos superiores a `manual-full` são
destacados no relatório final.

## Resultados e limites da interpretação

As maiores médias de mAP@0.5:0.95 entre os volumes sintéticos no CitDet foram
0,240 para YOLOv8s, 0,243 para YOLO26s e 0,212 para RT-DETR-L. As referências
`manual-full` correspondentes foram 0,214, 0,236 e 0,161. Esses máximos foram
identificados após comparar cinco volumes no próprio CitDet; não equivalem a
uma escolha prévia de condição nem demonstram superioridade estatística.

![Médias de mAP em função do volume sintético no CitDet e na validação manual](docs/figures/results/synthetic-volume-vs-map.svg)

*Figura 2. Médias das duas sementes de treinamento. Linhas tracejadas indicam
`manual-full` para o mesmo detector. O eixo começa em zero, e `controlled`
aparece como referência pontilhada. A validação manual à direita foi usada
na seleção dos checkpoints de `manual-full`; não é um segundo teste externo.*

- O gerador foi calibrado com estatísticas de caixas e aparência dos conjuntos
  avaliados. O CitDet permanece externo à coleta, mas não é um teste intocado
  pelo desenvolvimento. Uma confirmação exige um novo conjunto reservado.
- `manual_full_val` reutiliza 26 imagens de validação de `manual-full`. Isso
  favorece a avaliação dessa condição por seleção de checkpoint; a magnitude
  do viés não foi estimada.
- O split sintético separa cenas, permitindo compartilhar fundos e recortes.
  A validação mede novas composições de ativos conhecidos.
- Há duas sementes de treinamento e um único pool sintético. As médias não
  estimam a variabilidade de outras coletas ou gerações independentes. Não
  foram publicados aqui intervalos de confiança ou testes de equivalência.
- Volumes maiores também trazem mais caixas, validações maiores e mais passos
  por época. O experimento não isola apenas a quantidade de imagens nem o
  efeito de cada transformação do gerador.

Os valores completos, MAE de contagem, exemplos de detecção e mapas de calor
estão em [`docs/RESULTS.md`](docs/RESULTS.md). A economia de trabalho humano e
uma margem aceitável de perda de desempenho ainda precisam ser medidas para
responder à pergunta de pesquisa.

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
| `scripts/diagrams/` | geração e exportação dos fluxogramas |
| `docs/DIAGRAMS.md` | instruções, origem das miniaturas e revisão visual |

Os resultados são salvos em `artifacts/confirmatory/`. O relatório final fica
em `artifacts/confirmatory/RESULTS_citdet.md`.

## Verificação local

```bash
.venv/bin/python -m pytest -q
uvx ruff check .
```
