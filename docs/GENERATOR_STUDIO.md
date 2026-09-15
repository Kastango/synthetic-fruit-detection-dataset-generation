# Studio

## Instalar e abrir

Use Python 3.11 ou 3.12 na raiz do repositório.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python scripts/studio.py
```

Abra [127.0.0.1:8765](http://127.0.0.1:8765). Para instalar o kit, clique em
**Baixar dados de demonstração**.
O pacote contém seis fundos, seus mapas de profundidade e 32 recortes RGBA.
O clone inclui o ZIP de cerca de 7 MB. Se faltar, o servidor tenta baixá-lo.
O Studio verifica tamanho e SHA-256 antes de extrair em `data/studio-demo`.

O kit contém versões reduzidas das fotografias do projeto para experimentar a
ferramenta. Os hashes das fontes e as transformações estão em `provenance.json`
no pacote. Consulte os [termos das imagens](../README.md#avaliação-e-fontes).

Para usar seus próprios ativos preparados:

```bash
.venv/bin/python scripts/studio.py --asset-root /caminho/ativos
```

A pasta deve conter `backgrounds`, `backgrounds_map` e `pictures_trimmed`.
Cada fundo deve ter seu mapa correspondente, e os recortes devem preservar
a transparência. Por padrão, o Studio prefere o catálogo completo
em `data/assets/regenerated` quando ele existe.

## Ajustar a receita

Comece pela quantidade e pelo tamanho das frutas. Ajuste a oclusão e a aparência
depois. A cada mudança, confira se as frutas cabem na copa, se a luz combina com
o fundo e se as bordas dos recortes continuam visíveis.

A receita inicial fica em [`studio.yaml`](../configs/synthesis/studio.yaml).
Ela usa os mesmos parâmetros de composição de `confirmatory_pool.yaml`,
com 390 imagens no total. Os controles da interface leem os valores do YAML.

O padrão solicita de 1 a 60 frutas com distribuição beta ajustada pela média
14,443 e variância 123,207 da mistura em partes iguais das duas coletas reais.
Rejeições e oclusões podem reduzir a quantidade final de caixas. Alterar o
mínimo ou o máximo reescala a distribuição. Limites iguais fixam a quantidade.
A receita reserva uma probabilidade independente de 1% para cenas negativas.

Use os parâmetros abaixo para modificar a composição:

| O que você quer mudar | Parâmetro | Efeito na cena |
|---|---|---|
| Repetir uma composição | `seed` | Repete os sorteios com os mesmos ativos, código e bibliotecas. O Studio começa com `42`. |
| Gerar mais imagens | `images.total` | Define o tamanho do pool. O Studio começa com 390 cenas. |
| Mudar o formato | `canvas` | Define largura e altura em pixels. `[720, 960]` produz retratos. |
| Mudar a quantidade de frutas | `objects.min`, `objects.max` | Sorteia de 1 a 60 com uma distribuição beta calibrada pela média e variância dos cenários reais. Rejeições e oclusões podem reduzir o total visível. |
| Espelhar os ativos | `augmentation.horizontal_flip` | Ativa 50% de chance de espelhamento horizontal por fundo e por fruta. O mapa acompanha o fundo. |
| Aproximar ou afastar as frutas | `objects.min_scale`, `objects.max_scale`, `objects.scene_scale.spread` | Sorteia o centro de escala por cena e a dispersão dos recortes ao redor dele. |
| Esconder mais fruta atrás das folhas | `placement.z_offset` | Valores mais negativos colocam a fruta atrás de regiões próximas do fundo. O mapa usa proximidade normalizada de 0 a 255. |
| Rejeitar frutas quase ocultas | `placement.min_visibility` | Exige 15% de superfície visível, também verificados depois das oclusões finais. |
| Evitar a parte inferior da foto | `placement.exclude_bottom_fraction` | O padrão 0 libera toda a altura da imagem. |
| Variar a maturação | `appearance.ripeness` | Altera o matiz de parte dos recortes maduros para verde-amarelado. |
| Combinar a fruta com a luz local | `appearance.hsv_cast` | Aproxima cor e luminosidade da fruta das do fundo. |
| Variar sol e sombra entre frutas | `appearance.exposure_jitter` | Multiplica a intensidade por um fator entre 0,82 e 1,27 nas instâncias afetadas. |
| Suavizar o encontro com as folhas | `occlusion.edge_blur`, `occlusion.edge_feather_radius` | Suaviza a máscara de oclusão e o contorno do recorte. |
| Escolher o que a caixa cobre | `annotation.mode` | `visible` cobre a parte visível. `amodal` inclui a parte oculta. A receita usa `visible`. |

O gerador sorteia fundos com reposição. O espelhamento de fundos e frutas usa
sorteios independentes. O mapa de profundidade acompanha o fundo.

## Sementes e reprodução

Mantenha a semente para comparar os ajustes nas mesmas cenas. A exportação
usa `sampling.mode: paired-v1`. Mudanças de aparência preservam os sorteios
de geometria. Alterar contagem, tamanho ou catálogo pode mudar as caixas.
Reproduzir pixels exige os mesmos ativos, código e bibliotecas.

Para baixar o YAML, clique em **Salvar receita**. Para exportar imagens e caixas
YOLO, clique em **Gerar dataset**, escolha o total de imagens e a proporção de
treino e baixe o ZIP. O pacote inclui a receita, os manifestos e os hashes.
O processamento usa CPU e salva os resultados em `artifacts/studio`.

## Acesso na rede local

```bash
.venv/bin/python scripts/studio.py --host 0.0.0.0
```

Acesse `http://IP-DO-SERVIDOR:8765` em outra máquina da mesma rede.
