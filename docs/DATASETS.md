# Conjuntos de dados e protocolo experimental

Este documento descreve somente os dados que participam do protocolo atual:
a base real anotada, a condição `controlled`, os cinco conjuntos sintéticos e a
coleta externa de avaliação. As contagens de imagens vêm da
configuração; as caixas sintéticas foram contadas no pool de 13/09/2026.
As medições completas estão em [DATASET_PROFILES.md](DATASET_PROFILES.md).

## Visão geral

| Conjunto | Papel | Treino | Validação | Teste | Caixas conhecidas |
|---|---|---:|---:|---:|---:|
| `manual-full` | referência com anotação humana | 104 | 26 | — | 2.093 |
| `controlled` | controle sem composição | 284 | 71 | — | 127 |
| `synthetic-1x` | síntese no tamanho da base real | 104 | 26 | — | 6.612 |
| `synthetic-2x` | síntese | 208 | 52 | — | 11.620 |
| `synthetic-3x` | síntese | 312 | 78 | — | 17.790 |
| `synthetic-5x` | síntese, volume intermediário | 520 | 130 | — | 29.966 |
| `synthetic-10x` | maior volume sintético previsto | 1.040 | 260 | — | 61.425 |
| `oranges_field` | avaliação externa/desenvolvimento | — | — | 1.243 | 15.893 |

**Protocolo de treino.** São 42 execuções: 7 condições × 3 detectores ×
2 sementes. Todas usam até 50 épocas, `patience: 30`, `imgsz: 960`,
`mosaic: 1.0`, `close_mosaic: 5` e as mesmas augmentações, sem congelamento.
YOLOv8s e YOLO26s usam SGD, `lr0: 0.01`, batch 8. RT-DETR-L usa AdamW,
`lr0: 0.0001`, batch 2 por limite de memória. Os ajustes são constantes entre
condições dentro de cada arquitetura; diferem entre arquiteturas.
A grade permanece interrompida durante a revisão dos dados.

Cada condição de treinamento possui sua própria validação. O melhor checkpoint
de cada execução é escolhido sem consultar o teste externo; somente após a
seleção ser congelada em `model_selection.json` o teste externo é preparado e
avaliado. Nenhuma imagem da coleta externa entra no treinamento. Entretanto, seu
protocolo de anotação, erros de detector e estatísticas já orientaram o
gerador. A comparação 50/50 solicitada em 13/09/2026 formaliza esse uso como
desenvolvimento. Congelar checkpoints antes de avaliar não torna essa coleta
um teste independente. Uma avaliação confirmatória exige outra coleta
intocada; dividir agora recortes já consultados não desfaz esse uso.

Todas as condições são convertidas para detecção YOLO com uma única classe,
`poncan`. Na coleta externa isso significa uma classe genérica de fruto
cítrico; as imagens dela não contêm poncã.

## Base real anotada (`manual-full`)

A referência supervisionada vem de `datanotation.zip`: 130 fotografias de
poncãs em pomar, com 2.093 caixas delimitadoras anotadas manualmente.

| Verificação do arquivo original | Resultado |
|---|---:|
| Integridade ZIP/CRC | válida |
| Imagens com rótulo correspondente | 130 |
| Caixas YOLO válidas | 2.093 |
| Fotografias com iPhone 13 mini | 82 |
| Fotografias com Google Pixel 6a | 48 |
| Duplicatas exatas | 0 |
| Duplicatas com dHash idêntico | 0 |
| Linhas de anotação inválidas | 0 |
| SHA-256 | `28308d791546a72deb2033e3c4fca6db1e830bf1108b5afe0c9db46eac2500e3` |

Na importação, `ImageOps.exif_transpose` materializa a orientação registrada
pela câmera. Isso apenas alinha os pixels à orientação em que as caixas foram
anotadas; não é augmentation nem reamostragem aleatória.

O split é determinístico, usa semente 42 e preserva aproximadamente a proporção
dos dois aparelhos:

| Split | Imagens | iPhone | Pixel | Caixas |
|---|---:|---:|---:|---:|
| treino | 104 | 66 | 38 | 1.642 |
| validação | 26 | 16 | 10 | 451 |

O manifesto `artifacts/real_split_confirmatory.json` congela os IDs, a origem e
as contagens. O ZIP não é versionado; a pipeline valida tamanho e SHA-256 antes
de importá-lo.

## Ativos de campo usados por `controlled` e pela síntese

Os ativos de geração são dois grupos distintos das 130 imagens anotadas:

- 127 fotos aproximadas de frutas, usadas como positivos controlados e como
  fonte dos recortes RGBA;
- 228 fotos de pomar sem fruta-alvo anotada, usadas como negativos controlados
  e como fundos das composições.

O pré-processamento produz:

1. um recorte por foto de fruta com IS-Net (`isnet-general-use`), preservando a
   caixa da máscara no referencial da foto original;
2. uma versão RGB normalizada de cada fundo;
3. um mapa de profundidade DepthPro para cada fundo, na resolução original;
4. um catálogo que associa os 228 fundos aos respectivos mapas e indexa os 127
   recortes.

O fingerprint do catálogo depende dos ativos e da versão do pré-processamento.
Consulte-o no manifesto da execução. Modelos e revisões ficam registrados em
`data/assets/regenerated/preprocess_manifest.json`, e a lista exata de ativos
fica em `data/assets/regenerated/asset_catalog.json`.

Esses arquivos de campo não possuem uma licença de redistribuição declarada de
forma inequívoca na origem. Por isso ficam fora do Git e o download exige
`--accept-data-terms`. A licença do repositório cobre o código, não concede
automaticamente direitos sobre as imagens.

## Condição `controlled`

`controlled` responde a uma pergunta específica: quanto o detector aprende com
os ativos de campo antes de qualquer composição de cena? Ela é uma condição de controle sem
composição. Não isola uma única transformação do gerador.

### Construção

- Cada uma das 127 fotos de fruta permanece como fotografia real completa.
- A caixa positiva é derivada da máscara IS-Net no sistema de coordenadas da
  foto, resultando em exatamente uma caixa por imagem positiva.
- Cada um dos 228 fundos permanece como fotografia real e recebe um arquivo de
  rótulo vazio, funcionando como exemplo negativo.
- Frutas e fundos são particionados separadamente com proporção 80/20 e semente
  42. Assim, nenhum arquivo-fonte aparece nos dois splits de `controlled`.
- Na materialização dessa condição não há colagem, mapa de profundidade, sombra
  artificial, rotação ou mudança de escala. As augmentations aplicadas durante
  o treinamento continuam sendo as mesmas das demais condições.

| Split | Positivos de fruta | Fundos negativos | Total | Caixas |
|---|---:|---:|---:|---:|
| treino | 102 | 182 | 284 | 102 |
| validação | 25 | 46 | 71 | 25 |

O manifesto `artifacts/controlled_split.json` congela o fingerprint das fontes
e essas partições.

### Como interpretar

A condição mede o valor dos ativos e dos rótulos derivados automaticamente sem
o compositor. Ela também expõe o detector a uma proporção alta de negativos
(228 de 355 imagens), útil para medir falsos positivos em folhagem e galhos.

Ela não reproduz a tarefa final: as fotos positivas são close-ups com uma fruta,
enquanto as coletas reais contêm cenas de copa com muitas
instâncias, oclusões e escalas. A comparação com os conjuntos sintéticos altera
simultaneamente composição,
escala, pose, densidade de objetos e número de caixas. Não permite atribuir
o desempenho a apenas um desses fatores.

## Conjuntos sintéticos

### Como uma cena é formada

O gerador cria primeiro um pool único de 1.300 identidades de cena em resolução
720×960. Para cada identidade, uma semente estável determina toda a composição:

1. escolhe um par fundo/mapa de profundidade do catálogo;
2. corrige contraste, brilho e nitidez do fundo antes de inserir frutas;
3. em 7% das cenas para aqui, e a copa fica sem fruta nenhuma;
4. nas demais, solicita 10–110 frutas com distribuição em U;
5. sorteia um centro de escala log-uniforme por cena, varia cada fruta
   em torno dele (`spread: 2.24`) e sorteia sua rotação;
6. tenta posições, define a oclusão pelo eixo z e verifica a visibilidade
   mínima na inserção;
7. ajusta matiz, HSV cast e exposição, aplica sombras e compõe a cena em
   ordem de profundidade;
8. recompõe até que toda fruta desenhada renda uma caixa verificável, e
   extrai cada caixa da parte visível final;
9. salva imagem, rótulo YOLO e metadados da cena.

Parâmetros que definem o pool confirmatório:

| Propriedade | Valor atual |
|---|---|
| resolução gerada | 720×960, retrato |
| cenas sem fruta | probabilidade 7%; observado 84/1.300 = 6,46% |
| objetos solicitados nas demais | 10–110, curva em U, sem bloco `dense` |
| escala por cena | centro 0,012–0,055 do menor lado do canvas, log-uniforme; dispersão 2,24 |
| rotação | até ±180° |
| visibilidade mínima da fruta | 20%, verificada sobre a cena final |
| piso de fruta visível | 90 pixels de máscara |
| região inferior excluída da colocação | nenhuma |
| caixas | parte visível final, largura e altura mínimas de 2 px |
| split do pool | 80/20, semente 42 |
| qualidade JPEG | 95, sem subamostragem de croma |

A configuração integral e versionável está em
`configs/synthesis/confirmatory_pool.yaml`. As contagens descrevem o pool medido, gerado antes da simplificação do código; a
divisão por split e a cobertura dos ativos precisam ser consultadas nos
manifestos da execução, não inferidas da proporção de imagens.

### Formação de `synthetic-1x` a `synthetic-10x`

O pool é dividido uma única vez em 1.040 cenas de treino e 260 de validação.
Depois disso, cada condição toma prefixos crescentes de ambas as partições:

| Condição | Imagens treino | Imagens validação | Caixas totais publicadas |
|---|---:|---:|---:|
| `synthetic-1x` | 104 | 26 | 6.612 |
| `synthetic-2x` | 208 | 52 | 11.620 |
| `synthetic-3x` | 312 | 78 | 17.790 |
| `synthetic-5x` | 520 | 130 | 29.966 |
| `synthetic-10x` | 1.040 | 260 | 61.425 |

Os conjuntos são estritamente aninhados: `2x` contém todas as cenas de `1x`,
`3x` contém todas as de `2x` e assim por diante, tanto no treino quanto na
validação. Isso controla a identidade das cenas compartilhadas entre volumes. Não isola
apenas o número de imagens, pois validação, caixas e passos por época variam.

A identidade da cena é definida antes do split. Mudar apenas a proporção
treino/validação não altera seus pixels ou rótulos. Cada registro em
`manifest.jsonl` informa semente, fundo, mapa de profundidade, recortes,
quantidade solicitada/inserida, rejeições e caminhos de saída. A pipeline também
congela o hash da configuração, o fingerprint do catálogo e o SHA-256 do
manifesto do pool.

### Limites da síntese

- O split é feito por **identidade de cena**, não por ativo-fonte. Os mesmos
  recortes e fundos podem reaparecer em cenas diferentes de treino e validação;
  portanto, a validação sintética mede generalização para novas composições dos
  ativos conhecidos, não para um novo pomar.
- Solicitar frutas não garante uma caixa final. A frequência de cenas negativas
  deve ser consultada em `summary.json`; `controlled` é uma execução separada
  e não acrescenta seus fundos negativos ao treino sintético.
- Profundidade, segmentação e caixas visíveis são estimativas automáticas e
  podem propagar erros sistemáticos.
- A aparência dos fundos disponíveis é mais difusa/nublada e em ângulo distinto
  da base manual. A correção tonal modifica o fundo antes da
  composição; seu efeito isolado no desempenho não foi medido.
- Como as condições são aninhadas, resultados entre tamanhos são comparáveis;
  porém, não estimam a variância que seria obtida com pools sintéticos gerados
  independentemente.

## Coleta externa de avaliação (`oranges_field`)

A avaliação externa vem de [Carella et al. (2026)](https://doi.org/10.1016/j.compag.2026.111833),
publicada em [Mendeley Data](https://data.mendeley.com/datasets/93f32zgkxz/1)
sob CC BY-NC 3.0. São 865 fotografias de laranja doce (*Citrus sinensis*) em
pomares experimentais e comerciais da Sicília, tiradas com sete modelos de
celular, sem pose, distância ou ângulo fixos, entre outubro e dezembro de 2024.

### Por que ela substituiu a coleta anterior

A coleta externa usada antes anotava fruta na árvore e fruta no chão. No split
de teste, 6.263 das 10.082 caixas eram de chão — 62% do gabarito era uma
categoria que a coleta própria não contém e que o gerador não produz. Filtrar
o gabarito não resolve, porque as imagens continuam tendo fruta caída e uma
detecção correta dela passaria a contar como falso positivo.

### O que ela traz

- **Protocolo de anotação escrito.** Uma caixa por fruto; fruto com mais de
  aproximadamente 80% de oclusão não é anotado; fruto pequeno ou borrado demais
  para identificação segura é descartado. É o primeiro piso de visibilidade
  documentado contra o qual calibrar o gerador.
- **Nove condições de tempo e hora** — manhã, tarde e noite × sol, nublado e
  chuva, mais noturno e interior — e cinco cultivares em estágios do verde ao
  maduro.
- **Sete aparelhos diferentes**, o que espalha nitidez, faixa dinâmica e
  balanço de branco.

### O recorte que a pipeline usa

O pacote publicado não é o conjunto de teste: são 5.025 sub-imagens 640×640
extraídas das 865 fotos por um algoritmo de corte, e o corte introduz dois
problemas.

| condição | recortes | caixas | caixas com lado > 0,15 |
|---|---:|---:|---:|
| `MS` manhã ensolarada | 2.768 | 14.765 | 23,8% |
| `AC` tarde nublada | 1.082 | 14.445 | 5,6% |
| `AS` tarde ensolarada | 625 | 7.791 | 8,9% |
| `ES` fim de tarde ensolarado | 224 | 1.980 | 2,3% |
| `MC` manhã nublada | 126 | 2.784 | 0,0% |
| `NI` noturna | 96 | 862 | 3,5% |
| `IN` interior | 57 | 97 | 95,9% |
| `EC` fim de tarde nublado | 34 | 150 | 6,7% |
| `AR` tarde chuvosa | 13 | 164 | 1,2% |

Primeiro, parte dos recortes fica tão perto que uma fruta ocupa o quadro
inteiro: a pergunta ali não é "onde estão as frutas nesta copa", é "isto é uma
fruta". Segundo, as condições estão muito desbalanceadas, e só a de manhã
ensolarada responde por 55% dos recortes — justamente a mais contaminada por
recorte de aproximação.

`scripts/curate_oranges_field.py` aplica dois critérios, nenhum deles olhando
para a distribuição da coleta própria:

1. descarta o recorte sem caixas ou que contenha qualquer caixa com lado
   acima de 1/4 da imagem;
2. limita cada condição a 250 recortes, percorrendo as fotos de origem em
   rodadas para não pegar 250 recortes da mesma árvore;
3. exclui a condição de interior, que não é pomar.

| Verificação | Resultado esperado |
|---|---:|
| tamanho do pacote | 812.649.381 bytes |
| SHA-256 | `5400227ae218eacd6f060b982f83c4d56258bef66779a46dc0ff8b770b298e35` |
| recortes publicados | 5.025 |
| recortes no teste após a curadoria | 1.243 |
| fotos de origem representadas | 655 |
| caixas no teste | 15.893 |
| média de caixas por imagem | 12,8 |
| formato original das caixas | YOLO (COCO também disponível) |
| licença | CC BY-NC 3.0 |

A curadoria foi definida antes da comparação com a coleta própria. As
extensões normalizadas das caixas e a densidade medidas são:

| | p5 | mediana | p95 | caixas/imagem |
|---|---:|---:|---:|---:|
| `oranges_field` | 0,0187 | 0,0437 | 0,1281 | 12,8 |
| `manual-full` | 0,0201 | 0,0365 | 0,0893 | 16,1 |

### Como interpretar o resultado externo

Esta coleta é de outra equipe, outro país e outra espécie de citros, mas
seu protocolo, erros e estatísticas foram consultados durante o desenvolvimento.
Ela mede desempenho em um domínio externo conhecido; não é um teste intocado.

Três limites precisam acompanhar qualquer número dela:

- **A anotação é semiautomática.** Pré-rotulagem por um YOLOv8n treinado
  iterativamente, refinada à mão no Roboflow e filtrada por CLIP. Avaliar um
  detector da família YOLO contra rótulos derivados de YOLO tem risco de viés a
  favor da família. Auditar uma amostra com `scripts/audit_dataset.py` é o
  contrapeso.
- **Fruta caída é anotada na mesma classe.** Ao contrário da coleta anterior,
  aqui não há categoria separada, então a fração não é mensurável a partir do
  gabarito e não é possível filtrá-la. A inspeção visual indica que é bem menor
  que os 62% da coleta anterior, mas não é zero.
- **Os recortes são correlacionados.** A medição atual também encontrou dez
  repetições exatas por SHA-256 e cinco caixas duplicadas. Os casos estão no
  relatório de perfis; nada foi removido silenciosamente do gabarito.
  As 1.243 imagens vêm de 655 fotos, então
  o tamanho efetivo da amostra é menor que a contagem de imagens sugere.

Após o colapso de classes, a métrica é detecção genérica de fruto cítrico, não
reconhecimento taxonômico de poncã.

## Receita de síntese e protocolo de treino

A receita é [`configs/synthesis/confirmatory_pool.yaml`](../configs/synthesis/confirmatory_pool.yaml),
única fonte dos conjuntos sintéticos.

### Receita oficial e Studio

A receita calibrada pelo usuário é o padrão de ambos. Os YAMLs diferem somente
no nome e no total de imagens: Studio 390, pool 1.300. Os sliders leem o YAML;
um teste verifica que abrir/exportar os padrões preserva todos os parâmetros.
Foram preservados 25% de maturação, 12% de podridão, visibilidade 20%, piso de
90 pixels, z −7, chão liberado, luz por cena, espelhamento 50% e gradação com
variação por cena. A mistura 50/50 é um alvo de análise, ainda não uma nova
receita promovida. Consulte [DATASET_PROFILES.md](DATASET_PROFILES.md).

Foram removidas as opções experimentais `objects.depth_scale`,
`objects.dense.scale_with_count` e `placement.require_vegetation`. Receitas
que as contenham agora falham com orientação de migração, em vez de serem
interpretadas silenciosamente de outra forma. `scene_scale` continua ativo.

### Reprodutibilidade

O pool medido tem 1.300 imagens, 61.425 caixas e 84 negativos. Os manifestos
conservam a receita e o hash do gerador que realmente o produziu. A simplificação
altera o hash do código; a pipeline recusará retomar esse diretório sem
regeneração explícita. Não se deve reescrever o hash antigo para simular uma
nova geração. Seis cenas selecionadas por quantis, incluindo negativa e densa,
saíram com pixels e rótulos idênticos antes/depois da simplificação. Essa
verificação é amostral; não equivale à regeneração integral do pool.

O treino continua sem congelamento. As rodadas exploratórias e a grade
aposentada foram apagadas a pedido do usuário; seus números não constituem
resultados atuais. Foram preservadas as fontes, o pool oficial e a revisão
humana de anotações.

## Rastreabilidade

| Artefato | O que congela |
|---|---|
| `configs/pipeline.yaml` | fontes, hashes, tamanhos esperados, splits e caminhos |
| `configs/synthesis/confirmatory_pool.yaml` | protocolo completo de composição |
| `artifacts/real_split_confirmatory.json` | IDs e contagens da base manual |
| `artifacts/controlled_split.json` | partição dos positivos e negativos controlados |
| `data/assets/regenerated/preprocess_manifest.json` | modelos e produtos do pré-processamento |
| `data/assets/regenerated/asset_catalog.json` | pares fundo/profundidade e recortes elegíveis |
| `data/generated/confirmatory_pool/manifest.jsonl` | proveniência de cada cena sintética |
| `data/generated/*/summary.json` | tamanho e vínculo de cada subconjunto ao pool |
| `data/external_tests/oranges_field/manifest.json` | origem, hashes e conversão do teste externo |

Os diretórios `data/`, `artifacts/` e `runs/` são produtos locais ou dados
restritos e não substituem os manifestos versionados/configurações que definem
o protocolo. Uma reprodução válida deve passar pela validação da pipeline, não
apenas possuir arquivos com os mesmos nomes.
