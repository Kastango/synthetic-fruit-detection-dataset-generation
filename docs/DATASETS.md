# Conjuntos de dados e protocolo experimental

Este documento descreve somente os dados que participam do protocolo atual:
a base real anotada, a condição `controlled`, os cinco conjuntos sintéticos e o
CitDet como avaliação em coleta externa. As contagens de imagens vêm da
configuração; as caixas sintéticas totais são as publicadas em
[RESULTS.md](RESULTS.md). A auditoria histórica da base manual é preservada abaixo.

## Visão geral

| Conjunto | Papel | Treino | Validação | Teste | Caixas conhecidas |
|---|---|---:|---:|---:|---:|
| `manual-full` | referência com anotação humana | 104 | 26 | — | 2.093 |
| `controlled` | controle sem composição | 284 | 71 | — | 127 |
| `synthetic-1x` | síntese no tamanho da base real | 104 | 26 | — | 8.302 |
| `synthetic-2x` | síntese | 208 | 52 | — | 18.084 |
| `synthetic-3x` | síntese | 312 | 78 | — | 26.853 |
| `synthetic-5x` | síntese, volume intermediário | 520 | 130 | — | 44.284 |
| `synthetic-10x` | maior volume sintético avaliado | 1.040 | 260 | — | 91.623 |
| CitDet | teste externo comum | — | — | 119 | 10.082 |

Cada condição de treinamento possui sua própria validação. O melhor checkpoint
de cada execução é escolhido sem consultar o CitDet; somente após a seleção ser
congelada em `model_selection.json` o teste externo é preparado e avaliado. O
split de treino do CitDet não é usado no treinamento dos detectores. Esse
controle não impede calibração do gerador com estatísticas da avaliação,
limitação presente nesta versão.

Todas as condições são convertidas para detecção YOLO com uma única classe,
`poncan`. No CitDet, isso significa colapsar as categorias de localização do
dataset original em uma classe genérica de fruto cítrico; não significa que as
119 imagens contenham exclusivamente poncãs.

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
enquanto as fotos reais anotadas e o CitDet contêm cenas de copa com muitas
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
3. solicita 1–30 frutas ou, com probabilidade de 50%, 60–200 frutas;
4. sorteia escala-base e rotação do recorte;
5. tenta posições, usa a proximidade local para ajustar escala e oclusão e
   verifica a visibilidade mínima na inserção;
6. ajusta matiz, HSV cast e exposição, aplica sombras e compõe a fruta,
   atualizando a oclusão das instâncias anteriores;
7. após todas as inserções, extrai cada caixa da parte visível final;
8. salva imagem, rótulo YOLO e metadados da cena.

Parâmetros que definem o pool confirmatório:

| Propriedade | Valor atual |
|---|---|
| resolução gerada | 720×960, retrato |
| objetos solicitados por cena | 1–30 ou 60–200, com 50% de probabilidade para cada faixa |
| escala-base | maior lado do recorte = 0,01–0,065 do menor lado do canvas |
| rotação | até ±180° |
| escala guiada por profundidade | 0,6× (longe) a 1,3× (perto) |
| visibilidade mínima na inserção | 15% |
| região inferior excluída da colocação | 15% |
| caixas | parte visível final, largura e altura mínimas de 2 px |
| split do pool | 80/20, semente 42 |
| qualidade JPEG | 95, sem subamostragem de croma |

A configuração integral e versionável está em
`configs/synthesis/confirmatory_pool.yaml`. As contagens abaixo são as publicadas
em [`RESULTS.md`](RESULTS.md), agregando
treino e validação. O pool avaliado contém 91.623 caixas; as contagens da versão
anterior, de 19.934 caixas, não descrevem o gerador atual. A divisão de caixas
por split e a cobertura dos ativos precisam ser consultadas nos manifestos da
execução, não inferidas da proporção de imagens.

### Formação de `synthetic-1x` a `synthetic-10x`

O pool é dividido uma única vez em 1.040 cenas de treino e 260 de validação.
Depois disso, cada condição toma prefixos crescentes de ambas as partições:

| Condição | Imagens treino | Imagens validação | Caixas totais publicadas |
|---|---:|---:|---:|
| `synthetic-1x` | 104 | 26 | 8.302 |
| `synthetic-2x` | 208 | 52 | 18.084 |
| `synthetic-3x` | 312 | 78 | 26.853 |
| `synthetic-5x` | 520 | 130 | 44.284 |
| `synthetic-10x` | 1.040 | 260 | 91.623 |

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

## CitDet como teste externo

O [CitDet](https://robotic-vision-lab.github.io/citdet/) é um benchmark de
detecção de frutos cítricos em pomar publicado por James et al. no *IEEE
Robotics and Automation Letters* ([artigo](https://doi.org/10.1109/LRA.2024.3474473),
[dataset](https://doi.org/10.32855/dataset.2024.05.005)). A coleção completa
possui 579 imagens de alta resolução e mais de 32 mil caixas.

### Por que ele é adequado para este projeto

- Foi capturado em um pomar real, em Fort Pierce, Flórida, entre outubro de 2021
  e outubro de 2022.
- Contém mais de 60 variedades de cítricos afetadas por Huanglongbing (HLB), em
  diferentes estágios de maturação, cores e tamanhos.
- As árvores foram fotografadas em orientação retrato, de ambos os lados da
  fileira, em condições ensolaradas e sombreadas e ao longo de vários dias.
- A câmera foi posicionada próxima ao solo para simular a observação por um robô
  terrestre entre fileiras.
- Frutos na árvore e no chão são anotados, produzindo cenas densas e oclusas.

Essas propriedades tornam o CitDet um teste de transferência de domínio muito
mais exigente que uma separação aleatória das 130 fotos locais. Ele mede se as
características aprendidas com poncãs e/ou síntese se transferem para outros
cítricos, outra fazenda, outras condições ambientais e maior densidade de
objetos.

### Uso exato pela pipeline

A pipeline espera o pacote `UTA_CSE_Dataset.zip`, verifica sua identidade e
abre somente o arquivo interno `CitDet-test.zip`:

| Verificação | Resultado esperado |
|---|---:|
| tamanho do pacote | 1.103.158.596 bytes |
| SHA-256 | `15610a71de5540baf23f70b6c66123c30859ce42e0846dc843c21d277bfe71b1` |
| imagens do split oficial de teste | 119 |
| caixas do split oficial de teste | 10.082 |
| média de caixas por imagem | 84,72 |
| formato original das caixas | COCO JSON |
| licença do pacote CitDet | CC BY-NC-SA 4.0 |

As imagens são copiadas sem resize, recompressão ou alteração geométrica. As
caixas COCO são convertidas para YOLO e todas as categorias de fruto são
colapsadas para a classe 0. As pseudo-máscaras PNG disponibilizadas pelos
autores não são usadas, porque o protocolo avalia detecção por caixas.

O portal pode impor um desafio WAF a downloads não interativos. Nesse caso,
`--external-source` aceita uma cópia obtida manualmente, mas tamanho e SHA-256
continuam obrigatórios. O arquivo e o teste materializado permanecem fora do
Git por tamanho e licença.

### Como interpretar o resultado externo

Nesta versão, estatísticas do CitDet orientaram ajustes do gerador. O conjunto
é externo à coleta local, mas não foi mantido intocado durante o desenvolvimento.
Esse uso limita conclusões confirmatórias mesmo que suas imagens não entrem
no treinamento dos detectores.

O CitDet não mede reconhecimento taxonômico de poncã: após o colapso de classes,
mede detecção genérica de fruto cítrico. Seu alto número de objetos pequenos por
imagem também torna a métrica sensível a resolução de entrada, `max_det` e
oclusão. Por isso, o resultado deve ser lido como robustez fora do domínio, não
como substituto da validação local nem como estimativa direta de desempenho em
qualquer pomar brasileiro.

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
| `data/external_tests/citdet/manifest.json` | origem, hashes e conversão do teste externo |

Os diretórios `data/`, `artifacts/` e `runs/` são produtos locais ou dados
restritos e não substituem os manifestos versionados/configurações que definem
o protocolo. Uma reprodução válida deve passar pela validação da pipeline, não
apenas possuir arquivos com os mesmos nomes.
