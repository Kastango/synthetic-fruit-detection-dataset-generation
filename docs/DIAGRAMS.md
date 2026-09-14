# Editar as figuras do README

Os geradores Python são a fonte editável dos fluxogramas. Os SVGs publicados
embutem fotos e fontes e abrem sem rede. Não edite o SVG exportado à mão, pois
a próxima geração sobrescreve essas mudanças.

## Alterar texto, cores ou disposição

Na raiz do repositório, com Python 3.11 ou 3.12:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[diagrams]'
.venv/bin/python scripts/diagrams/build_flowchart.py
.venv/bin/python scripts/diagrams/build_conditions.py
.venv/bin/python scripts/diagrams/check.py
```

Se `.venv` já existe, use o ambiente existente. A instalação inicial requer
rede; a geração dos fluxogramas usa apenas os arquivos versionados e não
requer GPU, datasets completos, uma skill instalada ou a antiga pasta externa
`diagrama/`. Os caminhos internos são relativos ao repositório, independentemente
da pasta em que o script é chamado.

| Arquivo | O que editar |
|---|---|
| [`build_flowchart.py`](../scripts/diagrams/build_flowchart.py) | `panel_acquisition`, `panel_generation` e `combined_svg` definem nós, rótulos, setas e zonas. As constantes iniciais definem cores, fontes e espaçamento. |
| [`build_conditions.py`](../scripts/diagrams/build_conditions.py) | `row_extent`, `CARD_W`, `CARD_H` e `GUTTER` controlam o tamanho das miniaturas e os espaços entre conjuntos. |
| [`fontkit.py`](../scripts/diagrams/fontkit.py) | Incorpora somente os glifos usados, a partir das fontes locais. Falha se faltar um glifo. |
| [`diagram-assets/`](figures/diagram-assets/) | Miniaturas e registros de origem usados nas figuras. |
| [`fonts/`](figures/fonts/) | Geist, Geist Mono e Instrument Serif, com as respectivas licenças SIL OFL. |

As contagens das condições vêm de `configs/pipeline.yaml`. O fluxograma lê
densidade, escala, rotação e limiares de `configs/synthesis/confirmatory_pool.yaml`.
A geometria acomoda os cinco multiplicadores atuais; adicionar condições ou
alongar rótulos exige rever a disposição. Alterações na ordem das operações
precisam ser conferidas contra `fruit_pipeline/synthesis.py`.

Saídas:

- `docs/figures/fluxograma-geracao-conjuntos-sinteticos.svg`;
- `docs/figures/condicoes/condicao-*.svg`, um arquivo por condição;
- `artifacts/diagrams/fluxograma-geracao-conjuntos-sinteticos.html`, preview local.

O fluxograma usa rótulos em inglês e mostra as etapas do compositor.
Treino e validação usam duas cartas por 26 imagens, com arredondamento. O
baralho do teste é idêntico nas sete condições, então seu tamanho não compara
nada: ele tem teto de 22 cartas e quem informa o volume é o rótulo abaixo.
O espaçamento diminui nas pilhas maiores; a largura não é uma escala linear. Os SVGs têm 1.116 px de largura.
Treino, validação e teste começam no mesmo X em todas as condições.
As setas ocupam o intervalo entre o fim de cada pilha e o grupo seguinte.
Os rótulos dos diagramas estão em inglês.
As setas entre treino e validação representam a avaliação entre épocas.
A validação não atualiza os pesos.

## Atualizar as miniaturas

Mudar o desenho usa as miniaturas versionadas. Regenerar as fotos requer os
dados locais preparados pela pipeline. Para as pilhas e faixas do fluxograma:

```bash
.venv/bin/python scripts/diagrams/build_assets.py \
  --preview data/generated/confirmatory_pool
```

O argumento deve apontar para um dataset com `images/train`, `labels/train`
e `metadata/train`. O script também usa `data/raw/fruits` e
`data/assets/regenerated/{backgrounds,backgrounds_map,pictures_trimmed}`.
As três fotos de frutas têm recortes correspondentes; as três árvores têm seus
próprios mapas DepthPro `_depth.png`. As cinco faixas do laço usam a mesma
tentativa de inserção, com oclusão parcial escolhida para tornar o processo
visível. A faixa de escala e giro amplia o recorte para facilitar a leitura;
sua dimensão desenhada não representa a escala no canvas. Essa seleção é
didática, não uma amostra aleatória de qualidade.

Para atualizar os microfluxogramas a partir dos splits materializados:

```bash
.venv/bin/python scripts/diagrams/build_conditions_assets.py \
  --dataset manual=data/real_yolo_confirmatory \
  --dataset controlled=data/real_controlled \
  --dataset synthetic=data/generated/confirmatory_pool \
  --dataset test=data/external_tests/oranges_field
```

Cada `--dataset` é independente. O script amostra arquivos em ordem estável,
intercala positivos e negativos de `controlled` usando seus rótulos e registra
nome e SHA-256 das imagens em `cell_sources.json`. Repita os comandos de build
e verificação após mudar as miniaturas.

### Proveniência das miniaturas publicadas

As miniaturas das condições vêm dos conjuntos materializados em disco:
`data/real_yolo_confirmatory`, `data/real_controlled`,
`data/generated/confirmatory_pool` e `data/external_tests/oranges_field`.
`cell_sources.json` registra nome e SHA-256 de cada cena mostrada;
`provenance.json` registra a configuração do fluxograma e a inserção ilustrada.
Elas representam tipos de dado e não certificam a identidade dos arquivos de
cada split nem constituem evidência de desempenho.

As faixas do laço do fluxograma vêm de um preview ilustrativo, gerado com a
receita vigente e `images.total` reduzido. Como o total integra o hash da
configuração, esse preview não é um prefixo do pool. Para reproduzi-lo com os
ativos já preparados:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import yaml
directory = Path('artifacts/diagrams')
directory.mkdir(parents=True, exist_ok=True)
config = yaml.safe_load(Path('configs/synthesis/confirmatory_pool.yaml').read_text())
config['images']['total'] = 80
(directory / 'preview.yaml').write_text(yaml.safe_dump(config))
PY
.venv/bin/python scripts/generate_synthetic.py \
  --synthesis-config artifacts/diagrams/preview.yaml \
  --output artifacts/diagrams/preview-80 --workers 4
.venv/bin/python scripts/diagrams/build_assets.py --preview artifacts/diagrams/preview-80
.venv/bin/python scripts/diagrams/build_conditions_assets.py \
  --dataset synthetic=artifacts/diagrams/preview-80
```

As imagens de campo seguem os termos registrados em [DATASETS.md](DATASETS.md).
As miniaturas do teste externo são recortes e redimensionamentos da coleta de
[Carella et al.](https://data.mendeley.com/datasets/93f32zgkxz/1), sob
[CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/).
As fontes vieram de [Google Fonts](https://github.com/google/fonts/tree/main/ofl),
nos diretórios `geist`, `geistmono` e `instrumentserif`; as licenças acompanham
os arquivos. A licença do código não substitui os termos desses recursos.

## Gráfico e exemplos de resultados

```bash
.venv/bin/python scripts/plot_confirmatory_results.py
```

Esse comando lê os JSONs por execução em `artifacts/confirmatory`, gerados pela
avaliação. Gera quatro gráficos de tendência (mAP, precision, recall, F1), uma
folha de ranking por conjunto de teste e tabelas intermediárias, mas não
reescreve a interpretação em `RESULTS.md`. Detector sem resultado é ignorado,
então a grade pode ser rodada uma arquitetura de cada vez e consolidada depois.

Os exemplos de detecção usam `scripts/render_detection_examples.py`, que
requer checkpoints e o conjunto escolhido preparado. Use `--dataset`,
`--model`, `--image-stem` e `--seed`; `--device cpu` permite executar sem GPU.
Os comandos e exemplos sintéticos estão em [RESULTS.md](RESULTS.md).
Preserve a mesma imagem e limiar entre condições e informe
esses valores na legenda. Uma imagem escolhida para ilustração não demonstra
desempenho médio.

## Conferir e exportar

Abra o HTML de preview em um navegador e confira também o README renderizado.
O verificador estrutural não detecta sobreposição visual. Antes de versionar:

- Confira textos, setas, pontas, legendas e margens na largura do README.
- Confirme que as fotos carregam e que as faixas usam a mesma inserção.
- Verifique proximidade de 8 bits, mínimo de caixa por lado, grading só no
  fundo e extração das caixas depois de todas as inserções.
- Mantenha explícita a distinção entre ilustração, validação e avaliação.
- Rode `git diff --check` e revise os scripts junto dos SVGs gerados.

Para extrair um SVG do HTML ou exportar PNG opcionalmente:

```bash
.venv/bin/python scripts/diagrams/export_diagram.py \
  artifacts/diagrams/fluxograma-geracao-conjuntos-sinteticos.html --format svg

# Somente para exportar PNG, instale o renderizador uma vez:
.venv/bin/python -m pip install playwright
.venv/bin/python -m playwright install chromium
.venv/bin/python scripts/diagrams/export_diagram.py \
  artifacts/diagrams/fluxograma-geracao-conjuntos-sinteticos.html --format png --scale 2
```

As exportações ficam junto do HTML em `artifacts/diagrams/`. O README usa os
SVGs de `docs/figures/`; não é necessário versionar os previews HTML ou PNG.
