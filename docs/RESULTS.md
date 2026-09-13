# Resultados dos detectores e exemplos de dados

**Não há grade publicada nesta página.** Duas coisas mudaram ao mesmo tempo e
nenhum número anterior sobrevive às duas:

1. **O gerador.** Toda fruta composta passou a exigir um rótulo que uma pessoa
   consiga verificar na imagem final, a cena é composta em ordem de
   profundidade, e a escala de cada cena passou a ter uma distância de câmera.
   Os conjuntos sintéticos mudaram.
2. **O conjunto de avaliação externa.** O CitDet saiu do protocolo.

Publicar a tabela antiga com ressalva seria pior do que não publicá-la: ela
descreveria um gerador que não existe contra um gabarito que não usamos mais.

## Por que o CitDet saiu

O split de teste do CitDet tem 10.082 caixas, das quais **6.263 (62%) são fruta
no chão** e 3.819 são fruta em árvore. Um detector treinado em cenas de copa
estava sendo medido majoritariamente contra fruta caída — categoria que a
coleta própria não contém e que o gerador não produz. A inspeção dos erros
confirma: as caixas perdidas concentram-se em fruta apodrecida sobre
serapilheira.

Filtrar o gabarito para `Fruit on Tree` não resolve. As imagens continuam
contendo fruta no chão, e como as detecções sem par contam como falso
positivo, o detector passa a ser punido por acertar. Isso troca um viés por
outro em vez de remover algum.

## Conjuntos de avaliação

| Conjunto | Composição | Papel e limite |
|---|---|---|
| `oranges_field` | 1.243 imagens e 15.893 caixas, curadas de [Carella et al. (2026)](https://data.mendeley.com/datasets/93f32zgkxz/1): laranja doce em árvore, sete celulares, oito combinações de hora e tempo | Avaliação externa. Outra equipe, outro país, outra espécie de citros; não participou de nenhum ajuste do gerador. O critério de curadoria está em [DATASETS.md](DATASETS.md#coleta-externa-de-avaliação-oranges_field). |
| `manual_full_val` | 26 imagens e 451 caixas da validação de `manual-full` | Avaliação local. Essas imagens selecionam os checkpoints de `manual-full`, então essa condição não tem estimativa independente aqui. |

O `oranges_field` traz o que faltava: um **protocolo de anotação escrito**. Uma
caixa por fruto; fruto com mais de aproximadamente 80% de oclusão não é
anotado; fruto pequeno ou borrado demais para identificação segura é
descartado. Isso dá um piso de visibilidade documentado contra o qual calibrar
o gerador, em vez de um número escolhido por inspeção.

Três ressalvas que precisam acompanhar qualquer número dele:

- A anotação é semiautomática — pré-rotulagem por um YOLOv8n treinado
  iterativamente, refinada à mão no Roboflow e filtrada por CLIP. Avaliar um
  detector da família YOLO contra rótulos derivados de YOLO tem risco de viés
  a favor da família.
- **Fruta caída também é anotada**, na mesma classe única. Não há categoria
  separada, então a fração não sai do gabarito e não dá para filtrar. A
  inspeção visual indica que é bem menor que os 62% da coleta anterior, mas não
  é zero.
- As imagens são recortes 640×640 de fotos maiores: as 1.243 do teste vêm de
  655 fotos de origem, então o tamanho efetivo da amostra é menor que a
  contagem sugere.

O viés de seleção do `manual_full_val` foi medido comparando o checkpoint
escolhido com o último de cada execução: **+0,0039** de mAP@.50:.95 a favor de
`manual-full`, contra +0,0004 nas condições sintéticas, que selecionam na
própria validação sintética.

## Primeira leitura no conjunto novo

Checkpoints que já existiam, avaliados no `oranges_field` sem retreinar. Duas
sementes de treino por condição, YOLOv8s. Não é uma grade — é a verificação de
que o conjunto novo distingue o que o anterior não distinguia.

| condição | `oranges_field` | coleta anterior | `manual_full_val` |
|---|---:|---:|---:|
| receita calibrada à mão | 0,2215 | 0,2207 | 0,3728 |
| ↳ com distância de câmera | **0,2594** | 0,2220 | 0,3856 |
| ↳ com distância e teto maior | **0,2691** | 0,2186 | 0,3886 |
| receita de referência | 0,2819 | 0,2257 | 0,3949 |
| ↳ com distância de câmera | **0,2951** | 0,2281 | 0,3946 |

A distância de câmera por cena rende **+0,0379** e **+0,0132** nas duas bases,
com as duas sementes concordando nos dois casos. Na coleta anterior os mesmos
pares rendiam +0,0013 e +0,0024 — indistinguíveis do ruído entre sementes, que
é de 0,001 a 0,005 aqui. O conjunto anterior era praticamente cego a um
mecanismo cujo efeito o novo mede com folga, e a ordem das condições agora
coincide com a da avaliação local.

## Protocolo de treino

Continua o de [`configs/confirmatory.yaml`](../configs/confirmatory.yaml):
7 condições × 3 detectores × 2 sementes, 42 execuções que compartilham 50
épocas, `imgsz` 960, `batch` 8, `freeze: 5`, `mosaic` 1.0 com `close_mosaic` 5
e as mesmas augmentações de cor. Nenhum hiperparâmetro varia por condição ou
por detector.

## Garantias que o gerador entrega

Estas valem sobre a cena final, depois de todas as oclusões, e são verificadas
por teste:

| Garantia | Valor |
|---|---|
| Fruta desenhada sem rótulo | nenhuma |
| Visibilidade mínima da fruta entregue | a que a receita declara, medida sobre a fruta e não sobre a caixa |
| Lado mínimo da caixa | o que a receita declara |
| Oclusão invertida na imagem entregue (fruta ao fundo cobrindo fruta à frente) | nenhuma |

A escala é calibrada contra a distribuição real de caixas, e a dispersão de
tamanho segue a da fotografia: pouca variação dentro de uma cena (p90/p10 ≈
1,8×) e muita entre cenas (≈ 2,4×), contra 1,8× e 2,6× medidos no
`manual-full`.

## Histórico dos experimentos exploratórios

`artifacts/historico_experimentos.csv` reúne, numa linha por condição, o
mAP@.50:.95 médio e o desvio de cada mecanismo já triado. Serve de registro,
não de resultado: são triagens de duas a quatro sementes de treino sobre um ou
dois sorteios do gerador.

Duas medidas de ruído delimitam o que essas triagens conseguem afirmar. Entre
sementes de treino da mesma receita, o desvio típico é de 0,003 a 0,005 de
mAP. Entre dois sorteios do gerador com a mesma receita, a diferença chega a
0,015 — maior que quase todo efeito de receita já medido. Por isso duas
sementes triam e quatro promovem, e por isso um mecanismo só entra quando o
argumento físico se sustenta sozinho.

## Exemplos de dados sintéticos

Cenas geradas com semente raiz 42. A seleção usa os quantis 25%, 50%, 75% e
97% da contagem de caixas, com desempate pelo índice de geração; não usa
resultados de detector nem seleção estética. As cenas sem caixas permitem
inspecionar problemas de inserção, escala e iluminação que as métricas não
resumem.

[![Quatro cenas sintéticas com seus gabaritos automáticos](figures/results/sheets/cenas-sinteticas.jpg)](figures/results/sheets/cenas-sinteticas.jpg)

Reproduza a exportação com `.venv/bin/python scripts/render_synthetic_examples.py`.
O dataset deve estar gerado conforme o [guia do estúdio](GENERATOR_STUDIO.md).
O [registro de origem](figures/results/synthetic-examples/provenance.json)
preserva sementes por cena, hashes, índices e a regra de seleção.

## Auditar um gabarito

Antes de confiar num conjunto de avaliação, vale olhar as imagens. A auditoria
mostra uma por vez com as caixas desenhadas e grava o veredito por tecla:

```bash
.venv/bin/python scripts/audit_dataset.py \
  --images data/external_tests/<nome>/images/test \
  --labels data/external_tests/<nome>/labels/test \
  --output artifacts/auditoria/<nome>.jsonl \
  --sample 60 --checkpoint runs/<...>/weights/best.pt
```

Com `--checkpoint`, a fila começa pelas imagens em que o detector e o gabarito
mais discordam, que é onde o erro de anotação se concentra. `--resumo` conta as
marcas depois. A ferramenta registra veredito; não altera caixa nenhuma.

## Como reproduzir

```bash
.venv/bin/python scripts/download_external.py oranges_field
unzip -q data/archives/oranges-in-the-field.zip -d /tmp/oranges
.venv/bin/python scripts/curate_oranges_field.py \
  --source "/tmp/oranges/Oranges in the field" --output /tmp/oranges_curado
./run_pipeline.sh prepare-test --external-name oranges_field \
  --external-source /tmp/oranges_curado
./run_pipeline.sh test --device 0 --unlock-test --external-name oranges_field
./run_pipeline.sh test --device 0 --unlock-test --external-name manual_full_val
./run_pipeline.sh report --external-name oranges_field
./run_pipeline.sh report --external-name manual_full_val
.venv/bin/python scripts/plot_confirmatory_results.py
```

Os JSONs por execução ficam em `artifacts/confirmatory`, fora do Git, e
acompanham o gerador vigente.

Os 42 checkpoints publicados como asset de release **precedem tudo isto** e não
correspondem a nenhuma tabela desta página nem ao conjunto de avaliação atual.
Ficam disponíveis apenas para reexecutar a avaliação do gerador anterior:
`.venv/bin/python scripts/download_weights.py`, validado por SHA-256 em
[`configs/pipeline.yaml`](../configs/pipeline.yaml).

## Rastreabilidade

- O bloqueio da avaliação após `model_selection.json` não desfaz o uso prévio
  de estatísticas da coleta própria no gerador nem o reuso da validação manual
  para seleção.
- Cada execução registra `dataset_fingerprint`, e cada pool registra
  `config_hash`, `asset_catalog_fingerprint` e `generator_sha256`. Conferir
  esses campos contra o disco é o primeiro passo para saber se uma tabela ainda
  descreve o gerador vigente.
