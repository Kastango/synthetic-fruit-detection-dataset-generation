# Detecção de poncãs com dados sintéticos

> Imagens sintéticas anotadas automaticamente podem reduzir o esforço de
> rotulagem e manter desempenho próximo ao obtido com imagens reais?

O projeto combina fotos de árvores sem frutas com recortes de poncãs. Dessa
combinação saem cenas de pomar e as caixas delas, sem anotação manual. O
experimento treina detectores nessas cenas e nas fotos de campo, e compara os
dois em duas coletas reais.

## Como as cenas são geradas

O DepthPro estima a profundidade de cada foto de árvore. O compositor usa esse
mapa para esconder parte das frutas atrás da vegetação. O tamanho da fruta varia
ao redor de um centro sorteado por cena, que representa a distância da câmera.
Depois da última inserção, o compositor recompõe as frutas do fundo para a
frente e verifica os pisos de 15% de visibilidade, 60 pixels visíveis de máscara
e 2 pixels por lado da caixa. Frutas que não passam são removidas da imagem;
a cena é recomposta e verificada novamente antes de extrair as caixas.

O ramo de 1% representa a probabilidade de solicitar uma cena sem frutas:
ele pula a composição de frutas e salva o fundo com um TXT vazio. Rejeições
no ramo de 99% podem reduzir a quantidade sorteada e também produzir cenas
vazias. O split de cenas é fixado antes da geração; ao final, os arquivos são
reunidos em subconjuntos aninhados.

[![Processo de preparação dos ativos, composição e divisão dos dados](docs/figures/fluxograma-geracao-conjuntos-sinteticos.svg)](docs/figures/fluxograma-geracao-conjuntos-sinteticos.svg)

Abra a figura em tamanho cheio para ler os parâmetros de cada etapa. As faixas
de cenas nela vêm de um preview ilustrativo, gerado com a receita vigente, e não
do pool publicado. O gerador parte de fotos RGB, sem sensor de profundidade e
sem modelagem 3D.

## Experimento

O experimento compara sete condições de treinamento. Cada conjunto sintético
contém o anterior, no treino e na validação.

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

As pilhas ilustram o volume de dados. Os rótulos dão a quantidade de imagens.

A avaliação local também usa as 26 imagens de `manual_full_val`. As setas entre
treino e validação representam a medição entre épocas. A validação não atualiza
pesos.

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
| YOLOs | SGD, taxa inicial 0,01, batch 8 |
| RT-DETR | AdamW, taxa inicial 0,0001, batch 2 |

Todos partem de pesos pré-treinados e ajustam todas as camadas, sem congelamento.

Cada treino escolhe seu checkpoint pela validação da própria condição. O
relatório então mede mAP, precision, recall e F1 nos dois conjuntos reais. Como
`manual_full_val` também participa da seleção de `manual-full`, seus resultados
favorecem essa condição. A coleta externa mede transferência para outro domínio,
mas também orientou a calibração do gerador. Uma conclusão confirmatória exige
uma terceira coleta, intocada pelo desenvolvimento.

## Resultados, e o que eles não provam

O gerador exige que toda fruta composta renda um rótulo que uma pessoa consiga
verificar na imagem. Nenhuma fruta é desenhada sem anotação. Nenhum rótulo fica
abaixo dos pisos que a receita declara. A verificação vale sobre a cena final,
depois de todas as oclusões, e não só no momento da inserção.

Na coleta externa, treinar em cena composta supera treinar nas 104 fotos
anotadas à mão nas três arquiteturas. São 0,193 contra 0,149 de mAP@.50:.95 no
YOLOv8s, 0,197 contra 0,149 no RT-DETR-L e 0,193 contra 0,174 no YOLO26s. Na
validação da coleta própria o quadro inverte, e por margem maior. O dado real
abre de 0,161 a 0,198, mas ali ele é avaliado no mesmo pomar, com os mesmos tipos de dispositivo e
com o mesmo anotador do seu treino. As maiores médias externas aparecem em 3x ou 5x, dependendo do detector.
Aumentar para 10x não melhora essas médias. São apenas duas sementes por
condição. Ainda precisamos de mais repetições e de uma coleta independente. As tabelas completas, os gráficos e os exemplos estão em
[docs/RESULTS.md](docs/RESULTS.md).

O que limita a leitura:

- A escala e a aparência da receita foram calibradas com as estatísticas de
  caixa da coleta própria. A coleta externa é de outra equipe, outro país e
  outra espécie de citro, e o protocolo dela também orientou o gerador. Ela é
  referência de desenvolvimento, sem independência confirmatória.
- `manual_full_val` reutiliza as 26 imagens de validação de `manual-full`. A
  seleção de checkpoint favorece essa condição nesse conjunto.
- O split sintético separa cenas, e fundos e recortes aparecem dos dois lados. A
  validação sintética mede composições novas de ativos conhecidos.
- São duas sementes de treino e um único pool sintético. As médias não estimam a
  variabilidade de outras coletas nem de gerações independentes. Não há
  intervalo de confiança nem teste de equivalência aqui.
- Volume maior traz também mais caixas, validação maior e mais passos por época.
  O experimento não isola a quantidade de imagens, nem o efeito de cada
  transformação do gerador.

Falta medir quanto trabalho humano a receita economiza, e qual perda de
desempenho é aceitável em troca. Sem esses dois números a pergunta de pesquisa
continua aberta.

## Preparar e executar

Use Python 3.11 ou 3.12. O treino exige GPU compatível com CUDA. O script cria
o ambiente virtual e instala as dependências.

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

Para retomar uma execução interrompida, repita o mesmo comando. A pipeline
reutiliza arquivos compatíveis e retoma cada treino pelo último checkpoint. Os
resultados ficam em `artifacts/confirmatory/`.

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

Abra [127.0.0.1:8765](http://127.0.0.1:8765). Se faltarem os ativos, clique em
"Baixar dados de demonstração". O kit traz seis fundos com profundidade e 32
recortes de fruta, e dispensa GPU e dados anotados. O pacote acompanha o clone.
Se ele estiver ausente, o Studio tenta baixá-lo.

O kit serve para experimentar a ferramenta. A grade de pesquisa usa o catálogo
completo, em `data/assets/regenerated`, que o Studio prefere quando existe. Para
apontar outro catálogo, acrescente `--asset-root /caminho/dos/ativos`.

1. Ajuste os sliders e confira as oito árvores preenchidas com poncãs.
2. Ative as caixas ou amplie os detalhes para conferir inserções e oclusões.
3. Mantenha a semente para comparar ajustes nas mesmas cenas.
4. Use "Salvar receita" para baixar o YAML.
5. Use "Gerar dataset", escolha o total de imagens e a proporção de treino e baixe o ZIP.

O ZIP traz imagens, caixas YOLO, a receita e os registros de reprodução. A
ferramenta roda em CPU e precisa apenas dos fundos, dos mapas de profundidade e
dos recortes já preparados. Você não precisa de um dataset real anotado.

Para abrir em outra máquina da rede, inicie com `--host 0.0.0.0` e acesse
`http://IP-DO-SERVIDOR:8765`. Veja os detalhes de
[sementes e exportação](docs/GENERATOR_STUDIO.md#sementes-e-reprodução).

<details>
<summary>Ajustar o gerador</summary>

Comece pela quantidade e pelo tamanho das frutas. Ajuste a oclusão e a
aparência depois. A cada mudança, confira se as frutas cabem na copa, se a luz
combina com o fundo e se as bordas dos recortes continuam visíveis.

A receita inicial do Studio fica em
[`studio.yaml`](configs/synthesis/studio.yaml).
Os caminhos abaixo identificam os campos do YAML.

| O que você quer mudar | Parâmetro | Efeito na cena |
|---|---|---|
| Repetir uma composição | `seed` | Repete os sorteios com os mesmos ativos, código e bibliotecas. O Studio começa com `42`. |
| Gerar mais imagens | `images.total` | Define o tamanho do pool. O Studio começa com 390 cenas. |
| Mudar o formato | `canvas` | Define largura e altura em pixels. `[720, 960]` produz retratos. |
| Mudar a quantidade de frutas | `objects.min`, `objects.max` | Sorteia de 1 a 60 com uma distribuição beta calibrada pela média e variância dos cenários reais. Rejeições e oclusões podem reduzir o total visível. |
| Espelhar os ativos | `augmentation.horizontal_flip` | Ativa 50% de chance de espelhamento horizontal por fundo e por fruta. O mapa acompanha o fundo. |
| Aproximar ou afastar as frutas | `objects.min_scale`, `objects.max_scale`, `objects.scene_scale.spread` | Sorteia o centro de escala por cena e a dispersão dos recortes ao redor dele. |
| Esconder mais fruta atrás das folhas | `placement.z_offset` | Valores mais negativos colocam a fruta atrás de regiões próximas do fundo. O mapa usa unidades de 0 a 255, não metros. |
| Rejeitar frutas quase ocultas | `placement.min_visibility` | Exige 15% de superfície visível, também verificados depois das oclusões finais. |
| Evitar a parte inferior da foto | `placement.exclude_bottom_fraction` | O padrão é 0: toda a altura está liberada. Não identifica o chão por segmentação. |
| Variar a maturação | `appearance.ripeness` | Altera o matiz de parte dos recortes maduros para verde-amarelado. |
| Combinar a fruta com a luz local | `appearance.hsv_cast` | Aproxima cor e luminosidade da fruta das do fundo. |
| Variar sol e sombra entre frutas | `appearance.exposure_jitter` | Multiplica a intensidade por um fator entre 0,82 e 1,27 nas instâncias afetadas. |
| Suavizar o encontro com as folhas | `occlusion.edge_blur`, `occlusion.edge_feather_radius` | Suaviza a máscara de oclusão e o contorno do recorte. |
| Escolher o que a caixa cobre | `annotation.mode` | `visible` cobre a parte visível. `amodal` inclui a parte oculta. A receita usa `visible`. |

A interface exporta `sampling.mode: paired-v1`. Nesse modo, mudar a aparência
preserva os sorteios de geometria. Os fundos são sorteados com reposição, e os
espelhamentos também saem da semente. Mudar quantidade, escala ou catálogo pode
alterar posições e caixas. Os manifestos registram semente e hash de cada
geração. A receita oficial, `confirmatory_pool.yaml`, usa a mesma composição do
Studio, com 1.300 imagens no total, 1% de cenas sem fruta e piso de 60 pixels
visíveis.

</details>
