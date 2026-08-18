# Auditoria e procedência dos dados

## Base real original

Fonte local auditada: `datanotation.zip`.

| Verificação | Resultado |
|---|---:|
| Integridade ZIP/CRC | válida |
| Imagens | 130 |
| Rótulos correspondentes | 130 |
| Caixas YOLO válidas | 2.093 |
| iPhone 13 mini | 82 |
| Google Pixel 6a | 48 |
| Resolução abaixo da câmera | 0 |
| Duplicatas exatas | 0 |
| Duplicatas com dHash idêntico | 0 |
| Linhas de anotação inválidas | 0 |

Os nomes (`IMG_*` e `PXL_*`) e os EXIF concordam integralmente sobre o aparelho.
O ZIP tem SHA-256
`28308d791546a72deb2033e3c4fca6db1e830bf1108b5afe0c9db46eac2500e3`.

Na importação, `ImageOps.exif_transpose` materializa a orientação usada durante
a anotação. Essa é uma normalização geométrica determinada pelo metadado da
câmera, não uma amostragem de augmentation. Os rótulos normalizados não são
alterados. O arquivo original e seu hash permanecem registrados no manifesto.

O split usa semente 42 e estratificação proporcional por aparelho:

| Split | Total | iPhone | Pixel | Caixas |
|---|---:|---:|---:|---:|
| treino | 100 | 64 | 36 | 1.628 |
| validação | 15 | 9 | 6 | 220 |
| teste | 15 | 9 | 6 | 245 |

O ZIP não inclui um manifesto de partição. A divisão 100/15/15 materializada
pela pipeline é determinística, estratificada por aparelho e adequada para
testes operacionais. Segundo o registro da coleta, imagens com nomes ou horários
próximos correspondem a árvores, sessões ou cenas suficientemente distintas.
Por isso, a imagem é tratada como unidade experimental, sem agrupamento inferido
a partir da proximidade temporal. Essa premissa deve permanecer declarada no
protocolo e no relatório do estudo.

## Exportação Roboflow excluída como fonte

`'Ponca 3 v2.zip` declara explicitamente:

- exportação Roboflow `ponca-3`, versão 2;
- auto-orientação e remoção do EXIF;
- resize 1280×1280 com preenchimento refletido;
- três versões das imagens de treino com exposição aleatória de −20% a +20%.

Sua estrutura tem 4.800 imagens em treino, 400 em validação e 130 em teste. As
5.200 primeiras são sintéticas; as 130 últimas são as fotos reais já
pré-processadas. Como mistura imagens geradas, transformações e fotos reais,
esse ZIP não entra como fonte canônica da pipeline.

Os rótulos da versão Roboflow não são idênticos aos 2.093 presentes em
`datanotation.zip`: ela contém caixas adicionais em muitas imagens. As duas
fontes devem ser tratadas como revisões de anotação distintas. A fonte
`datanotation.zip` permanece congelada, e validação e teste devem passar por
adjudicação antes da execução confirmatória.

## Ativos sintéticos públicos

O Google Drive referenciado pelo repositório contém um pacote preparado de
1.544.768.252 bytes. A pipeline extrai e utiliza somente:

- 228 imagens de fundo;
- mapas de profundidade preparados;
- 127 recortes de poncã.

Arquivos auxiliares presentes no pacote de origem não são extraídos nem entram
no inventário de ativos. Os arquivos brutos necessários permanecem públicos em
dois ZIPs separados: 127 fotos de frutas e 228 fundos. O notebook Colab
associado aos ativos usa DIS/IS-Net e ZoeDepth. Nenhum desses arquivos inclui
as 130 imagens reais com anotações.

A licença do projeto cobre o código, mas o repositório de origem não declara de
forma inequívoca a licença dos arquivos de campo. Por isso o download exige
`--accept-data-terms` e os dados ficam fora do Git.
