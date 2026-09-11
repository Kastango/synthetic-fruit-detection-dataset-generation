# Detecção de poncãs com dados sintéticos

> Imagens sintéticas anotadas automaticamente podem reduzir o esforço de
> rotulagem e manter desempenho próximo ao obtido com imagens reais?

O projeto combina fotos de árvores sem frutas com recortes de poncãs para
criar cenas de pomares e suas caixas automaticamente. O experimento compara
detectores treinados com esses dados aos treinados com fotografias de campo,
buscando desempenho próximo ao real e cenas verossímeis.

## Como as cenas são geradas

O DepthPro estima a profundidade das árvores. O compositor usa esses mapas
para ajustar o tamanho das frutas e esconder parte delas atrás da vegetação.
Depois de todas as inserções, calcula as caixas das partes visíveis.

[![Processo de preparação dos ativos, composição e divisão dos dados](docs/figures/fluxograma-geracao-conjuntos-sinteticos.svg)](docs/figures/fluxograma-geracao-conjuntos-sinteticos.svg)

Abra a figura para ler os detalhes. As miniaturas ilustram o processo.
O gerador usa fotos RGB, sem exigir sensor de profundidade ou modelagem 3D.

## Experimento

O experimento compara sete condições de treinamento. As pilhas crescem com
o número de imagens, e cada conjunto sintético contém o menor nas duas partições.

<table>
<thead><tr><th width="180">Condição</th><th>Treino, validação e teste</th></tr></thead>
<tbody>
<tr><td width="180"><code>manual-full</code></td><td>Fotografias de campo<br><br><img src="docs/figures/condicoes/condicao-manual-full.svg" alt="Conjuntos de manual-full, com pilhas proporcionais ao volume" width="1116"></td></tr>
<tr><td width="180"><code>controlled</code></td><td>Frutas isoladas e fundos negativos<br><br><img src="docs/figures/condicoes/condicao-controlled.svg" alt="Conjuntos de controlled, com pilhas proporcionais ao volume" width="1116"></td></tr>
<tr><td width="180"><code>synthetic-1x</code></td><td>Cenas sintéticas<br><br><img src="docs/figures/condicoes/condicao-synthetic-1x.svg" alt="Conjuntos de synthetic-1x, com pilhas proporcionais ao volume" width="1116"></td></tr>
<tr><td width="180"><code>synthetic-2x</code></td><td>Contém synthetic-1x<br><br><img src="docs/figures/condicoes/condicao-synthetic-2x.svg" alt="Conjuntos de synthetic-2x, com pilhas proporcionais ao volume" width="1116"></td></tr>
<tr><td width="180"><code>synthetic-3x</code></td><td>Contém synthetic-2x<br><br><img src="docs/figures/condicoes/condicao-synthetic-3x.svg" alt="Conjuntos de synthetic-3x, com pilhas proporcionais ao volume" width="1116"></td></tr>
<tr><td width="180"><code>synthetic-5x</code></td><td>Contém synthetic-3x<br><br><img src="docs/figures/condicoes/condicao-synthetic-5x.svg" alt="Conjuntos de synthetic-5x, com pilhas proporcionais ao volume" width="1116"></td></tr>
<tr><td width="180"><code>synthetic-10x</code></td><td>Contém synthetic-5x<br><br><img src="docs/figures/condicoes/condicao-synthetic-10x.svg" alt="Conjuntos de synthetic-10x, com pilhas proporcionais ao volume" width="1116"></td></tr>
</tbody>
</table>

As pilhas ilustram o volume de dados; os rótulos mostram a quantidade de imagens.

A avaliação local também usa as 26 imagens de `manual-full.val`.
As setas entre treino e validação representam a avaliação entre épocas,
sem atualização de pesos pela validação.

### Treino e avaliação

A grade contém **42 treinos**, sete condições, três detectores e duas sementes.
A configuração completa está em
[`confirmatory.yaml`](configs/confirmatory.yaml).

| Ajuste | Valor |
|---|---|
| Detectores | YOLOv8s, YOLO26s e RT-DETR-L |
| Sementes de treino | 41 e 42 |
| Duração | Até 50 épocas, com `patience: 30` |
| Entrada | `imgsz: 960` |
| Congelamento | `freeze: 5`, os cinco primeiros módulos de cada modelo |
| YOLOs | SGD, taxa inicial 0,01, batch 8 |
| RT-DETR | AdamW, taxa inicial 0,0001, batch 2 |

Todos partem de pesos pré-treinados. O congelamento reduz os parâmetros
ajustados, mas não o tamanho do detector. Os cinco módulos não representam
a mesma estrutura nas três arquiteturas.

Cada treino escolhe seu checkpoint pela validação da própria condição.
Depois, o relatório mede mAP, precision, recall e F1 nos dois conjuntos reais.
`manual-full.val` também participa da seleção de `manual-full`. O CitDet
orientou ajustes do gerador, portanto uma confirmação independente exige
outra coleta reservada.

## Resultados e limites da interpretação

**A grade está sendo regerada.** O gerador passou a exigir que toda fruta
composta renda um rótulo que uma pessoa consiga verificar na imagem: nenhuma
fruta é desenhada sem anotação, e nenhum rótulo fica abaixo dos pisos que a
receita declara. A verificação vale sobre a cena final, depois de todas as
oclusões, e não apenas no momento da inserção.

Isso muda os conjuntos sintéticos, então os números anteriores descreviam um
gerador que não existe mais e foram retirados em vez de mantidos com ressalva.
Os detalhes do protocolo e dos conjuntos de avaliação seguem em
[docs/RESULTS.md](docs/RESULTS.md).


- O gerador foi calibrado com estatísticas de caixas e aparência dos conjuntos
  avaliados. O CitDet permanece externo à coleta, mas não é um teste intocado
  pelo desenvolvimento. Uma confirmação exige um novo conjunto reservado.
- `manual_full_val` reutiliza 26 imagens de validação de `manual-full`. Isso
  favorece a avaliação dessa condição por seleção de checkpoint; o viés foi
  medido em +0,0039 de mAP@.50:.95, contra +0,0004 nas condições sintéticas.
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

## Preparar e executar

Use Python 3.11 ou 3.12. O treino requer GPU compatível com CUDA.
O script cria o ambiente virtual e instala as dependências.

Para preparar os dados e os ativos usados pela ferramenta:

```bash
./run_pipeline.sh prepare --device 0 --accept-data-terms
```

Para conferir a configuração sem iniciar os treinos:

```bash
./run_pipeline.sh all --dry-run --device 0 --accept-data-terms
```

Para executar a grade e liberar a avaliação após a seleção dos checkpoints:

```bash
./run_pipeline.sh all --device 0 --accept-data-terms --unlock-test
```

Repita o comando para retomar uma execução interrompida. A pipeline reutiliza
arquivos compatíveis e retoma treinos pelo último checkpoint.
Os resultados ficam em `artifacts/confirmatory/`.

As fontes, licenças, contagens e hashes estão em [DATASETS.md](docs/DATASETS.md).
Os caminhos de download ficam em [pipeline.yaml](configs/pipeline.yaml).

Para conferir o código localmente:

```bash
.venv/bin/python -m pytest -q
uvx ruff check .
```

## Visualizar e criar dados

Na raiz do clone, use Python 3.11 ou 3.12:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python scripts/studio.py
```

Abra [127.0.0.1:8765](http://127.0.0.1:8765). Se faltarem os ativos,
clique em "Baixar dados de demonstração". O kit inclui seis fundos com
profundidade e 32 recortes de frutas, sem precisar de GPU ou dados anotados.
O pacote acompanha o clone; se estiver ausente, o Studio tenta baixá-lo.

Esse kit serve para experimentar a ferramenta. A grade de pesquisa usa o
catálogo completo. O Studio usa `data/assets/regenerated` quando disponível.
Para indicar outro catálogo, acrescente `--asset-root /caminho/dos/ativos`.

1. Ajuste os sliders e confira as oito árvores preenchidas com poncãs.
2. Ative as caixas ou amplie os detalhes para conferir inserções e oclusões.
3. Mantenha a semente para comparar ajustes nas mesmas cenas.
4. Use "Salvar receita" para baixar o YAML.
5. Use "Gerar dataset", escolha o total de imagens e a proporção de treino e baixe o ZIP.

O ZIP inclui imagens, caixas YOLO, receita e registros de reprodução. A
ferramenta usa CPU e precisa apenas dos fundos, mapas de profundidade e
recortes preparados. O usuário não precisa de um dataset real anotado.

Para abrir em outra máquina da rede, inicie com `--host 0.0.0.0` e acesse
`http://IP-DO-SERVIDOR:8765`. Veja os detalhes de
[sementes e exportação](docs/GENERATOR_STUDIO.md#sementes-e-reprodução).

<details>
<summary>Ajustar o gerador</summary>

Comece pela quantidade e pelo tamanho das frutas. Depois ajuste a oclusão e
a aparência. Confira se as frutas cabem na copa, se a luz combina com o fundo
e se as bordas dos recortes continuam visíveis.

A receita inicial do Studio fica em
[`studio.yaml`](configs/synthesis/studio.yaml).
Os caminhos abaixo identificam os campos do YAML.

| O que você quer mudar | Parâmetro | Efeito na cena |
|---|---|---|
| Repetir uma composição | `seed` | Repete os sorteios com os mesmos ativos, código e bibliotecas. O Studio começa com `42`. |
| Gerar mais imagens | `images.total` | Define o tamanho do pool. O Studio começa com 390 cenas. |
| Mudar o formato | `canvas` | Define largura e altura em pixels. `[720, 960]` produz retratos. |
| Mudar a quantidade de frutas | `objects.min`, `objects.max` | Sorteia em uma curva em U. Quantidades próximas dos limites aparecem mais; próximas do centro, menos. Rejeições e oclusões podem reduzir o total visível. |
| Espelhar os ativos | `augmentation.horizontal_flip` | Ativa 50% de chance de espelhamento horizontal por fundo e por fruta. O mapa acompanha o fundo. |
| Aproximar ou afastar as frutas | `objects.min_scale`, `objects.max_scale`, `objects.depth_scale` | Controla o tamanho inicial do recorte e sua correção pela profundidade. |
| Esconder mais fruta atrás das folhas | `placement.z_offset` | Valores mais negativos colocam a fruta atrás de regiões próximas do fundo. O mapa usa unidades de 0 a 255, não metros. |
| Rejeitar frutas quase ocultas | `placement.min_visibility` | Exige uma fração visível na inserção. O valor oficial é 0,15. Outras frutas ainda podem cobri-la depois. |
| Evitar a parte inferior da foto | `placement.exclude_bottom_fraction` | Exclui os 15% inferiores do sorteio de posições. Não identifica o chão por segmentação. |
| Variar a maturação | `appearance.ripeness` | Altera o matiz de parte dos recortes maduros para verde-amarelado. |
| Combinar a fruta com a luz local | `appearance.hsv_cast` | Aproxima cor e luminosidade da fruta das do fundo. |
| Variar sol e sombra entre frutas | `appearance.exposure_jitter` | Multiplica a intensidade por um fator entre 0,40 e 1,90 nas instâncias afetadas. |
| Suavizar o encontro com as folhas | `occlusion.edge_blur`, `occlusion.edge_feather_radius` | Suaviza a máscara de oclusão e o contorno do recorte. |
| Escolher o que a caixa cobre | `annotation.mode` | `visible` cobre a parte visível. `amodal` inclui a parte oculta. A receita usa `visible`. |

A interface exporta `sampling.mode: paired-v1`. Nesse modo, mudar a aparência
preserva os sorteios de geometria. Fundos são sorteados com reposição.
Os espelhamentos também usam a semente.
Mudar quantidade, escala ou catálogo pode alterar posições e caixas.
Os manifestos registram sementes e hashes para identificar cada geração.
A receita histórica `confirmatory_pool.yaml` conserva os dois grupos usados
nos experimentos publicados. Esses resultados não avaliam a nova curva em U.

</details>
