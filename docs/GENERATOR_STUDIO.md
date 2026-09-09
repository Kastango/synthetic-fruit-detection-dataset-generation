# Estúdio e desenvolvimento do gerador

O objetivo é um detector treinado apenas com imagens sintéticas que funcione
nas duas coletas reais do projeto: o `manual-full`, anotado à mão, e o
[CitDet](https://robotic-vision-lab.github.io/citdet/), coleta externa. Esta
página descreve a ferramenta de composição, a receita oficial e as decisões
de protocolo que sustentam a grade de [RESULTS.md](RESULTS.md).

## Interface

```bash
.venv/bin/python scripts/studio.py
```

Abra http://127.0.0.1:8765. Para acesso na rede local:

```bash
.venv/bin/python scripts/studio.py --host 0.0.0.0
```

A página não depende de CDN, serviço externo, GPU ou dataset real anotado.
Precisa apenas dos ativos preparados: fundos de árvores, seus mapas de
profundidade e recortes de poncãs. Há 13 sliders, quatro cenas do próprio
compositor, caixas opcionais, detalhes ampliados e a opção de mostrar o
fundo original. A interface é uma ferramenta de criação e inspeção sintética;
os dados reais ficam apenas nos scripts de pesquisa.

“Salvar receita” baixa o YAML. “Gerar dataset” permite escolher a quantidade
de imagens e a fração destinada ao treino e cria um ZIP com imagens, rótulos
YOLO, `data.yaml`, `recipe.yaml`, manifestos, sementes e hashes dos ativos
e versões de bibliotecas em `provenance.json`. O servidor gera um
dataset por vez em CPU, exibe progresso e mantém os arquivos em
`artifacts/studio/datasets`. O ZIP não contém caminhos absolutos da máquina.
A mesma receita e quantidade reutilizam o trabalho concluído na sessão.

O ajuste dispara uma nova prévia após uma breve pausa; uma resposta antiga
não substitui os controles novos. O resumo da amostra fica recolhido e mostra
somente estatísticas das quatro cenas. Os detalhes ampliam os primeiros seis
rótulos de cada cena, sem seleção pelas predições de um detector.

A interface usa a receita oficial,
[`configs/synthesis/confirmatory_pool.yaml`](../configs/synthesis/confirmatory_pool.yaml).
Os 13 sliders restringem os ajustes de uso cotidiano; constantes mecânicas
continuam existindo e aparecem no YAML.

## Sementes e reprodução

A amostragem pareada é declarada na configuração:

```yaml
seed: 42
sampling:
  mode: paired-v1
```

Nesse modo a semente de cada cena depende da semente raiz, do conteúdo dos
ativos e do índice global da cena. O nome do experimento, o tamanho do pool e
os ajustes de aparência não trocam essa identidade. Há um sorteador por
instância e um fluxo separado para aparência: desligar exposição ou sombras
não consome números do sorteador de geometria. Alterações de geometria,
densidade ou catálogo naturalmente podem mudar as caixas e os objetos.

A CLI também aceita `--seed 123 --sampling-mode paired-v1`, sem editar o YAML.

O split possui uma semente separada (`synthetic_split.seed` em
`configs/pipeline.yaml`); mudar a quantidade de imagens pode mudar a atribuição
treino/validação, mas preserva a composição do mesmo `generation_index`.
As sementes 41/42 do treino são independentes da semente 42 da geração.

O YAML exportado pela interface pode ser usado diretamente:

```bash
.venv/bin/python scripts/generate_synthetic.py \
  --synthesis-config /caminho/studio_candidate.yaml \
  --output data/generated/studio_candidate --workers 6
```

A cena 1 da interface corresponde a `generation_index: 0`, localizada pelo
manifesto depois do split. JPG da interface é uma miniatura recodificada; a
imagem de treinamento original mantém o JPEG configurado. O compositor é o
mesmo em ambos.

Manifestos registram semente por cena, hash da configuração e SHA-256 do
compositor; o catálogo registra SHA-256 de cada ativo. O marcador de geração
recusa retomar um diretório produzido por outra configuração, código,
catálogo ou split. A mudança da aparência altera o manifesto mesmo quando as
caixas e posições são iguais, impedindo reutilização indevida dos treinos.
Reprodução de bytes também depende das mesmas versões de Python, NumPy e
Pillow. Treino determinístico de GPU tem limites próprios de hardware e
bibliotecas; não se promete equivalência de pesos entre máquinas diferentes.

## A receita oficial

A receita é [`configs/synthesis/confirmatory_pool.yaml`](../configs/synthesis/confirmatory_pool.yaml),
única fonte dos conjuntos sintéticos. O pool tem 1.300 imagens de 720×960, das
quais os subconjuntos `1x` a `10x` são recortes aninhados. A composição usa
dois regimes: cenas esparsas com 1 a 30 frutas e cenas densas com 60 a 110,
estas em 25% dos sorteios.

### Por que 25% de cenas densas

A escolha vem da distribuição de caixas por imagem das coletas reais:

| conjunto | imagens | média | p25 | mediana | p75 | p95 | máximo |
|---|---:|---:|---:|---:|---:|---:|---:|
| CitDet (teste) | 119 | 84,7 | 43 | 78 | 121 | 174 | 233 |
| `manual-full` (treino) | 104 | 15,8 | 9 | 14 | 22 | 31 | 40 |
| `manual-full` (validação) | 26 | 17,3 | 11 | 18 | 24 | 28 | 31 |
| pool sintético (25%) | 312 | 33,1 | 10 | 21 | 61 | 97 | 109 |

As duas coletas reais são regimes opostos: mediana 14 e máximo 40 no
`manual-full`, mediana 78 e máximo 233 no CitDet. Nenhuma distribuição
sintética única é verossímil para as duas ao mesmo tempo. Com poucas cenas
densas o gerador replica o `manual-full` e ignora o CitDet; a 25% ele cobre os
dois, com a metade inferior no regime do `manual-full` e o quartil superior
entrando na faixa central do CitDet. As cenas densas permanecem abaixo do
máximo real observado, que é 233 caixas. Aumentar a proporção de cenas densas
não afasta o gerador do campo: deixa de modelar apenas uma das duas realidades
coletadas.

### Congelamento uniforme do backbone

A grade usa `freeze: 5` em **todas** as condições e todos os modelos,
inclusive nos treinos com dados reais. É uma decisão de projeto tomada com o
custo já medido, e o custo fica registrado aqui em vez de omitido.

Congelar os blocos 0–4 prejudica o treino com dados reais. Medido com o mesmo
protocolo e as mesmas sementes, `manual-full` sem congelamento obtém 0,5459 no
conjunto local e com `freeze: 5` obtém 0,5062, uma perda de 0,0346, com as
duas sementes abaixo das do controle. No CitDet a diferença é nula, de 0,2126
para 0,2130. No treino sintético o mesmo corte rende entre +0,0088 e +0,0161
no CitDet sem custo local, replicado em três conjuntos e duas arquiteturas.

O motivo de aplicá-lo mesmo assim é a uniformidade: comparar condições exige
que treino, épocas, augmentação e congelamento sejam os mesmos para todas
elas. Um protocolo escolhido por condição, ainda que cada escolha fosse ótima
isoladamente, tornaria as diferenças entre condições ininterpretáveis.

Ao ler a grade, considere que a escolha do congelamento foi feita depois de
medir que ele favorece o treino sintético e desfavorece o real, e que portanto
a linha de base real aparece abaixo do seu próprio melhor desempenho possível.
Os números do treino real sem congelamento estão no parágrafo acima
justamente para que essa distância seja verificável.

A interpretação causal é que congelar remove a capacidade de adaptar filtros
de baixo nível à estatística do domínio de treino. Isso penaliza quem treina
no mesmo domínio em que testa e beneficia quem treina em cena composta:
o modelo congelado ajusta pior a validação sintética e transfere melhor,
o que é remoção de sobreajuste aos artefatos de composição, não ganho de
capacidade.

## Reprodutibilidade

O gerador foi regerado a partir da receita oficial em diretório separado:
390 imagens idênticas byte a byte, nenhum rótulo divergente, e `config_hash`,
`asset_catalog_fingerprint`, `generator_sha256` e `generator_schema_version`
coincidentes. O determinismo está verificado, não apenas alegado.

Manifestos registram semente por cena, hash da configuração e SHA-256 do
compositor; o catálogo registra SHA-256 de cada ativo. O marcador de geração
recusa retomar um diretório produzido por outra configuração, código, catálogo
ou split, o que impede que uma pasta seja sobrescrita depois de as corridas a
referenciarem. Reprodução byte a byte também depende das mesmas versões de
Python, NumPy e Pillow. Treino determinístico de GPU tem limites próprios de
hardware e bibliotecas; não se promete equivalência de pesos entre máquinas.

Os checkpoints são selecionados na validação sintética, nunca no conjunto de
teste. Uma candidata só avança como melhoria quando as médias melhoram nos
dois conjuntos de avaliação, e os resultados de ambas as sementes, regressões
individuais e a magnitude frente à variação do controle continuam explícitos.
Um ganho pequeno pede confirmação, não uma declaração de superioridade
estatística.

## Comparação com dados reais na pesquisa

Esta análise não faz parte da interface e não é necessária para gerar dados:

```bash
.venv/bin/python scripts/compare_similarity.py \
  --dataset data/generated/synthetic-3x \
  --output artifacts/studio/reference_similarity.json
```

O script compara todo o split sintético com o treino local (104 imagens) e o
CitDet já usado no desenvolvimento (119 imagens). Mede quantidade, escala,
posição, luminância, saturação, contraste e pixels claros saturados. A
distância é a média da diferença absoluta entre 101 quantis, nas unidades da
característica. Não há uma pontuação geral de realismo.

Aparência dentro de caixas inclui folhagem e oclusões. Essas medidas não
identificam toda colagem implausível, fruta solta ou contexto inadequado; a
inspeção visual continua necessária. Os resultados de detector e a similaridade
são reportados separadamente, por cenário.

Descritores agregados próximos não demonstram igualdade das distribuições
conjuntas ou da aparência. A fração de pixels laranja é um indicador
imperfeito: pode variar por maturação, sombra e cor do fundo, além de oclusão.

## Limites conhecidos e direções

As 2.093 caixas reais pertencem às 130 imagens do `manual-full`: 1.642 no
treino e 451 na validação. O catálogo mantém 127 recortes elegíveis.

Alguns fundos apresentam folhas menores e textura mais densa que as
referências de cítricos; uma cor semelhante não resolve isso. As direções
abertas partem de defeitos visíveis: bordas e luz dos recortes, contexto dos
fundos e pontos de inserção, ou a relação entre escala e densidade. Compare um
mecanismo por vez, com cenas pareadas e ao menos duas sementes, e exija
melhora conjunta nos dois conjuntos de avaliação.

## Direção visual

Árvores preenchidas com poncãs ocupam o centro; os 13 controles ficam à
esquerda e se reorganizam no celular. Fotos mantêm a proporção. Fundo,
caixas e detalhes são modos acionados pelo usuário, com estatísticas
recolhidas e sem comparação com dados reais na interface.
A paleta usa papel `#F6F8F7`, branco `#FFFFFF`, texto `#233A33`, folhagem
`#346851`, fruto `#EF9A35` e divisões `#D6DFDA`. A tipografia é Pomar Sans,
subconjunto local de Lato com licença incluída. Textos e títulos se limitam
ao necessário para ajustar, salvar e gerar dados.
