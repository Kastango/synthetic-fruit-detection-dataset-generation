# Detecção de poncãs com dados sintéticos

O projeto investiga se imagens sintéticas com anotação automática podem reduzir
o trabalho de rotulagem e manter desempenho próximo ao treino com fotos reais.
O gerador combina fotografias de pomar com recortes de poncãs, produz as caixas
delimitadoras e organiza os conjuntos usados no treinamento dos detectores.

A rodada publicada reúne 42 treinos. As maiores médias sintéticas de mAP superam
`manual-full` na coleta externa; na validação da coleta própria, `manual-full`
tem o maior mAP. A interpretação depende do domínio avaliado e do uso de cada
coleta durante o desenvolvimento. Os [resultados](docs/RESULTS.md) apresentam
as métricas, os exemplos e o alcance dessas conclusões.

## Geração das cenas

O IS-Net extrai os recortes das fotos de frutas. O DepthPro estima a
profundidade dos fundos a partir das fotos RGB. O compositor usa esses mapas
para posicionar as frutas e ocultar parte delas atrás da vegetação.

Cada cena recebe um centro de escala que representa a distância aparente da
câmera. O gerador varia tamanho, rotação, maturação, iluminação e sombras das
frutas. Depois, recompõe a cena em ordem de profundidade e extrai as caixas
da parte visível final. Cada instância deve conservar pelo menos 15% da máscara,
60 pixels visíveis e uma caixa com 2 pixels por lado. O compositor remove as
instâncias rejeitadas e repete a verificação.

A receita solicita de 1 a 60 frutas e reserva probabilidade de 1% para cenas
negativas. As rejeições podem reduzir a contagem final. A distribuição de
quantidade usa estatísticas das duas coletas reais; a escala usa a coleta de
poncãs. Esses ajustes fazem parte do desenvolvimento do gerador.

[![Preparação dos ativos, composição e divisão dos dados](docs/figures/fluxograma-geracao-conjuntos-sinteticos.svg)](docs/figures/fluxograma-geracao-conjuntos-sinteticos.svg)

As faixas de imagens do fluxograma ilustram as etapas do compositor. A
[receita de síntese](configs/synthesis/confirmatory_pool.yaml) define os
parâmetros e um pool de 1.300 cenas em 720 × 960 pixels.

## Dados e protocolo

`manual-full` contém 130 fotos de poncãs, com 2.130 caixas revisadas. A divisão
usa semente 42 e preserva aproximadamente a proporção de fotos dos aparelhos
utilizados, iPhone 13 mini e Google Pixel 6a. São 104 imagens e 1.667 caixas no
treino, e 26 imagens e 463 caixas na validação.

O catálogo de geração reúne outras 127 fotos de frutas e 228 fundos. A condição
`controlled` usa as fotos completas: uma caixa derivada da máscara IS-Net por
foto de fruta e um rótulo vazio por fundo. Frutas e fundos são divididos
separadamente, com 284 imagens no treino e 71 na validação.

O pool sintético tem 1.040 cenas de treino e 260 de validação. Os cinco volumes
usam subconjuntos aninhados em ambas as partições. Fundos e recortes são
compartilhados entre as partições, então a validação sintética avalia novas
composições de ativos conhecidos.

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

Os rótulos das figuras informam as contagens. As setas entre treino e validação
representam a avaliação entre épocas.

### Treinamento

A [configuração experimental](configs/confirmatory.yaml) combina sete condições,
três detectores e duas sementes, totalizando 42 treinos.

| Ajuste | Valor |
|---|---|
| Detectores | YOLOv8s, YOLO26s e RT-DETR-L |
| Sementes | 41 e 42 |
| Duração | Até 50 épocas, com `patience: 30` |
| Entrada | `imgsz: 960` |
| YOLOs | SGD, taxa inicial 0,01, batch 8 |
| RT-DETR-L | AdamW, taxa inicial 0,0001, batch 2 |

Todos partem de pesos pré-treinados e ajustam todas as camadas. Os parâmetros
permanecem fixos entre condições de uma mesma arquitetura. Cada execução
escolhe seu checkpoint pela validação da própria condição.

### Avaliação e fontes

A avaliação reúne dois conjuntos reais:

- `manual_full_val` contém as 26 fotos da validação própria. O uso dessas
  imagens na seleção de checkpoint de `manual-full` favorece essa condição.
- `oranges_field` contém 1.243 recortes e 15.893 caixas de laranja doce. Mede
  transferência para um domínio externo que também orientou a calibração do
  gerador. A comparação tem caráter exploratório.

O conjunto externo vem de [Oranges in the field](https://data.mendeley.com/datasets/93f32zgkxz/1),
com 5.025 imagens de 640 × 640 pixels e anotação semiautomática revisada por
pessoas. A [curadoria](scripts/curate_oranges_field.py) mantém recortes com
caixas de até 25% do lado da imagem, limita cada condição a 250 recortes e
exclui imagens de interior. A seleção percorre as fotos de origem em rodadas.
A avaliação usa uma classe de fruto cítrico.

Os endereços, tamanhos esperados e hashes dos pacotes estão em
[`pipeline.yaml`](configs/pipeline.yaml). O pacote manual inclui `review.jsonl`
e `CHECKSUMS.sha256`. Os manifestos gerados registram os splits, os ativos e a
configuração de cada conjunto.

A coleta externa usa [CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/).
O uso das fotografias próprias segue os termos dos respectivos pacotes, aceitos
com `--accept-data-terms`. A licença do código se aplica ao código.

## Resultados

Na coleta externa, as maiores médias de mAP@.50:.95 são 0,193 em `synthetic-5x`
para YOLOv8s, 0,197 em `synthetic-3x` para RT-DETR-L e 0,193 em `synthetic-5x`
para YOLO26s. As médias de `manual-full` são 0,149, 0,149 e 0,174,
respectivamente. No YOLO26s, 5x e 10x arredondam para o mesmo valor.

Na validação própria, `manual-full` supera a melhor média sintética de cada
detector por 0,161 a 0,198. Na coleta externa, também tem menor erro absoluto
médio de contagem que todas as condições sintéticas. A vantagem depende da
métrica e do conjunto.

A evidência se limita a duas sementes e um pool sintético. Aumentar o volume
altera também a quantidade de caixas, o tamanho da validação e os passos de
otimização. A economia de trabalho humano e a margem aceitável de perda de
desempenho permanecem questões de pesquisa. Veja a análise em
[RESULTS.md](docs/RESULTS.md).

## Preparar e executar

Use Python 3.11 ou 3.12 e uma GPU compatível com CUDA para o treinamento.
O script cria o ambiente virtual e instala as dependências.

Para preparar os dados e os ativos, execute:

```bash
./run_pipeline.sh prepare --device 0 --accept-data-terms
```

Para conferir o plano de execução, use:

```bash
./run_pipeline.sh all --dry-run --device 0 --accept-data-terms
```

Para treinar a grade e selecionar os checkpoints, execute:

```bash
./run_pipeline.sh train --device 0
./run_pipeline.sh select
```

Depois, siga a [preparação e avaliação dos conjuntos reais](docs/RESULTS.md#reproduzir-os-resultados).

Para retomar uma execução, repita o comando. A pipeline reutiliza os arquivos
compatíveis e retoma cada treino pelo último checkpoint. Os resultados ficam
em `artifacts/confirmatory/`. Os comandos para avaliar `manual_full_val`,
consolidar as tabelas e reproduzir as figuras estão em
[Reproduzir os resultados](docs/RESULTS.md#reproduzir-os-resultados).

Para conferir o código e a documentação, execute:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/check_documentation.py
uvx ruff check .
```

## Abrir o Studio

O Studio permite ajustar a receita, comparar cenas com a mesma semente e
exportar um dataset com imagens, caixas YOLO e registros de reprodução.
O processamento usa CPU e ativos já preparados.

Na raiz do repositório, execute:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python scripts/studio.py
```

Abra [127.0.0.1:8765](http://127.0.0.1:8765). Para instalar o kit incluído no
clone, clique em **Baixar dados de demonstração**. O pacote contém seis fundos
com mapas de profundidade e 32 recortes de fruta. Quando disponível, o Studio
prefere o catálogo completo em `data/assets/regenerated`.

Consulte o [guia do Studio](docs/GENERATOR_STUDIO.md) para ajustar parâmetros e
exportar dados, e o [guia das figuras](docs/DIAGRAMS.md) para editar os diagramas.
