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
| treino | 104 | 66 | 38 | 1.642 |
| validação | 26 | 16 | 10 | 451 |

O teste confirmatório é integralmente externo. O ZIP não inclui um manifesto de
partição; a pipeline materializa o split acima de forma determinística.

O ZIP manual não deve ser versionado. A pipeline registra a URL fornecida,
tamanho e SHA-256 em `configs/pipeline.yaml`, baixa o arquivo automaticamente e
valida sua identidade antes da importação. Um arquivo já disponível também pode
ser informado com `--real-source`.

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

## Teste externo

O teste padrão é o split oficial do
[CitDet](https://mavmatrix.uta.edu/cse_datasets/1/). O arquivo auditado contém
um ZIP de treino e outro de teste; a pipeline abre somente `CitDet-test.zip` e
converte suas caixas COCO para uma classe YOLO. As máscaras incluídas no pacote
não substituem as anotações de detecção.

| Verificação | Resultado |
|---|---:|
| ZIP externo | 1.103.158.596 bytes |
| SHA-256 | `15610a71de5540baf23f70b6c66123c30859ce42e0846dc843c21d277bfe71b1` |
| Imagens no teste | 119 |
| Caixas no teste | 10.082 |
| Formato de origem | COCO JSON |
| Licença declarada | CC BY-NC-SA 4.0 |

O portal oficial pode apresentar um desafio WAF para downloads não interativos.
O downloader tenta o endpoint oficial e, se necessário, aceita o mesmo ZIP por
`--external-source`. Tamanho e hash são verificados antes da extração. O teste
só pode ser materializado depois que `model_selection.json` existe.

O importador não é específico do CitDet: um diretório ou ZIP YOLO, COCO ou CVAT
pode ser registrado com outro nome. O conjunto
[Oranges in the field](https://data.mendeley.com/datasets/93f32zgkxz/1) é uma
alternativa pública com 5.025 imagens de 640×640 e licença CC BY-NC 3.0, mas seu
volume torna a preparação e a avaliação mais caras. O dataset descrito no
[artigo ELD-YOLO](https://www.mdpi.com/2223-7747/14/11/1729) não foi adotado:
o próprio artigo informa que as 2.388 imagens não estão disponíveis
publicamente e exigem contato com os autores.
