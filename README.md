# Pipeline experimental para detecção de poncãs com dados sintéticos

Este repositório implementa uma linha experimental reprodutível para estudar o
uso de imagens sintéticas no treinamento de detectores de poncãs. A pergunta
que orienta a pesquisa é:

> Sob um protocolo fixo de avaliação externa em imagens reais, dados sintéticos
> conseguem substituir imagens manualmente anotadas no treinamento de um
> detector de poncãs?

A pipeline preserva as fontes, reconstrói recortes e mapas de profundidade,
gera cenas parametrizáveis e executa grades YOLO em servidor. O teste real
permanece fechado enquanto decisões metodológicas, geradores, arquiteturas e
hiperparâmetros são definidos.

## Pergunta e hipótese principal

A pesquisa investiga substituição, não complementação. O experimento principal
não mistura fotografias reais anotadas com cenas sintéticas no treinamento.
Uma condição híbrida responderia a outra pergunta e, por isso, está fora deste
desenho.

A hipótese principal é que `synthetic-5x`, treinado exclusivamente com imagens
sintéticas, seja não inferior a `manual-full` no teste externo. `Synthetic-1x`,
`2x`, `3x` e `10x` formam uma curva de escala para mostrar se o desempenho
cresce, satura ou piora com o aumento do conjunto gerado. A condição
`controlled` separa o efeito de fotografar frutas reais em ambiente controlado
do efeito de compô-las em cenas sintéticas de pomar.

## Desenho experimental

As 130 imagens próprias anotadas serão divididas em 104 imagens de treino e 26
de validação, com estratificação por aparelho. Logo, `1x` corresponde a 104
imagens de treino. Cada fotografia de campo é uma unidade experimental: segundo
o registro da coleta, nomes ou horários próximos correspondem a árvores,
sessões ou cenas suficientemente distintas. Nas fotografias controladas, vistas
da mesma fruta física permanecem no mesmo lado da divisão quando essa identidade
puder ser recuperada.

| Família | Condição | Imagens de treino | Conteúdo |
|---|---|---:|---|
| manual | `manual-full` | 104 | fotografias de campo com caixas manuais |
| controlada | `controlled` | após auditoria | frutas controladas e folhagens negativas |
| sintética | `synthetic-1x` | 104 | somente cenas geradas |
| sintética | `synthetic-2x` | 208 | somente cenas geradas |
| sintética | `synthetic-3x` | 312 | somente cenas geradas |
| sintética | `synthetic-5x` | 520 | condição sintética principal |
| sintética | `synthetic-10x` | 1.040 | avaliação de saturação |

O gerador produzirá um pool mestre com 1.040 cenas de treino em dez blocos
balanceados de 104 e uma validação sintética fixa de 260 imagens. Os conjuntos
são aninhados: `2x` contém `1x`, `3x` contém `2x` e assim por diante. Dessa
forma, aumentar o volume não troca silenciosamente frutas, fundos ou parâmetros
por amostras mais favoráveis.

Todos os tratamentos serão executados com `yolo11s.pt` e `yolo26s.pt`, ambos
pré-treinados e com entrada 960. O split e o pool sintético usam semente de dados
42; as sementes de treinamento são 41 e 42. A matriz confirmatória contém:

```text
7 condições × 2 arquiteturas × 2 sementes = 28 treinamentos
```

As sementes são pareadas entre condições e a ordem das execuções é embaralhada.
Os dois resultados são apresentados individualmente, além da média: duas
sementes são o mínimo operacional e não estimam com precisão toda a variação
entre treinamentos.

O número de passos do otimizador, a política de augmentation, a regra de
checkpoint e os demais hiperparâmetros são mantidos fixos. Usar o mesmo número
de épocas seria uma análise diferente, pois daria mais atualizações às bases
maiores.

A augmentation é aplicada online e não aumenta a contagem `1x`–`10x`. Todas as
condições recebem a mesma política: variação HSV, translação, escala,
espelhamento horizontal e mosaic durante o treinamento. O mosaic é desligado no
último 10% das atualizações para estabilizar o ajuste final; as demais
transformações permanecem ativas, como no protocolo de referência. Rotação,
cisalhamento, perspectiva, espelhamento vertical, mixup e copy-paste ficam
desligados. Como as condições usam quantidades diferentes de imagens, a
pipeline converte os 10% finais no `close_mosaic` correspondente ao número de
épocas de cada execução.

## Avaliação externa e critério de substituição

O dataset externo anotado de poncãs será usado integralmente como teste. Nenhuma
de suas imagens poderá orientar treinamento, validação, escolha de parâmetros ou
construção dos ativos sintéticos. Os 28 modelos serão avaliados nas mesmas
imagens em uma única abertura planejada do teste.

Cada família seleciona checkpoints apenas com sua validação de origem: real para
`manual-full`, controlada para `controlled` e sintética para `synthetic-1x`–
`10x`. Usar a validação real para selecionar um detector sintético tornaria essa
condição assistida por anotações manuais e invalidaria a alegação de
substituição.

A métrica primária é mAP@0.5:0.95. Para `synthetic-5x`, calcula-se a diferença
em relação a `manual-full` com bootstrap pareado por imagem. Há evidência de
substituição quando o limite inferior do intervalo de confiança de 95% fica
acima de `-delta`, a margem prática que será congelada antes da avaliação.
Superioridade exige que esse limite seja maior que zero. A conclusão principal
deve se sustentar em YOLO11s e YOLO26s; resultado em apenas uma arquitetura será
descrito como dependente do modelo.

As métricas secundárias são mAP@0.5, precisão, revocação, F1, AP por tamanho e
erro de contagem por imagem. O limiar usado para precisão, revocação e F1 é
escolhido na validação de origem e congelado antes do teste. Caixas da mesma
imagem nunca são tratadas como observações estatísticas independentes.

## Fases e estado do estudo

1. **Auditoria dos dados: concluída.** As fontes reais e sintéticas foram
   identificadas, verificadas e registradas por hash.
2. **Piloto computacional: pronto.** Download, pré-processamento, síntese,
   treino, seleção e avaliação podem ser exercitados de ponta a ponta.
3. **Desenho confirmatório: definido.** Restam materializar as sete condições,
   escolher e auditar o dataset externo e congelar a margem de não inferioridade.
4. **Avaliação final: não iniciada.** O teste reservado não deve orientar
   alterações de método ou parâmetros.

## Estado dos dados

Dois ZIPs locais foram auditados:

- `datanotation.zip` é a base real original: 130 imagens, 130 rótulos YOLO,
  2.093 caixas, 82 fotos do iPhone 13 mini e 48 do Pixel 6a. Todas estão na
  resolução da câmera, não há duplicatas nem sinais de augmentation. SHA-256:
  `28308d791546a72deb2033e3c4fca6db1e830bf1108b5afe0c9db46eac2500e3`.
- `'Ponca 3 v2.zip` não é uma fonte bruta. É uma exportação Roboflow com
  resize 1280×1280, bordas refletidas e augmentation de exposição. Seus 5.330
  arquivos incluem 5.200 imagens sintéticas e 130 reais processadas.

O pacote público de ativos contém 228 fundos, seus mapas de profundidade e 127
recortes usados pela pipeline. Ele não contém as 130 imagens reais anotadas. Consulte
[docs/DATASETS.md](docs/DATASETS.md) para a auditoria e a procedência completas.

O dataset externo anotado ainda precisa ser escolhido e auditado. Até essa
etapa terminar, não existe resultado confirmatório nem conjunto final de teste
materializado na pipeline.

## Início rápido do piloto operacional

Python 3.11 ou 3.12 é necessário. O script cria e reutiliza `.venv`.

Os comandos abaixo validam a infraestrutura existente. A matriz confirmatória
com os cinco volumes sintéticos e as duas versões do YOLO ainda será
materializada em configurações próprias antes dos treinos finais.

```bash
# 1. Importar os originais e materializar o split piloto 100/15/15.
./run_pipeline.sh import-real --real-source ../datanotation.zip

# 2. Obter os ativos sintéticos já preparados (aprox. 1,4 GB).
./run_pipeline.sh download-prepared --accept-data-terms

# 3. Separar fundos e recortes de treino/validação e gerar uma configuração candidata.
./run_pipeline.sh split-assets
./run_pipeline.sh synthesize --workers 8 \
  --synthesis-config configs/synthesis/depth_robust.yaml

# 4. Auditar tudo antes de usar GPU.
./run_pipeline.sh validate
.venv/bin/python scripts/render_yolo_samples.py

# 5. Inspecionar a grade, depois treiná-la.
./run_pipeline.sh train --dry-run --device 0
./run_pipeline.sh train --device 0

# 6. Selecionar pela validação real. Este passo ainda não lê o teste.
./run_pipeline.sh select

# 7. Somente após congelar o protocolo confirmatório, abrir o teste uma vez.
./run_pipeline.sh test --device 0 --unlock-test
```

`configs/experiments.yaml` começa sem augmentation: `mosaic`, HSV, flip,
translação e escala estão zerados. Isso produz o baseline a partir das imagens
brutas do piloto. As configurações confirmatórias substituirão esses zeros pela
política fixa descrita acima; augmentation não será um eixo da grade. As
operações são aplicadas online pelo YOLO e nunca sobrescrevem
`data/real_source`.

## Reconstruir recortes e profundidade

O pacote pronto permite iniciar rapidamente. Para reproduzir o
pré-processamento desde as fotos brutas:

```bash
./run_pipeline.sh download-raw --accept-data-terms
./run_pipeline.sh preprocess --device 0

# Usar os recursos reconstruídos em vez do pacote preparado.
./run_pipeline.sh split-assets --asset-root data/assets/regenerated
./run_pipeline.sh synthesize --asset-root data/assets/regenerated \
  --workers 8 --synthesis-config configs/synthesis/depth_robust.yaml
```

A segmentação usa IS-Net/DIS por meio do `rembg`. A profundidade usa ZoeDepth
com revisão fixa do checkpoint. Os mapas são salvos em escala de cinza com as
regiões próximas em branco, como esperado pelo gerador.

## Configuração sintética piloto

`depth_robust.yaml` é a configuração candidata usada para validar a pipeline.
Ela combina escala relativa ao fundo, rotação de frutas, tentativas
limitadas, bordas de oclusão suavizadas e caixas calculadas sobre os pixels
visíveis.

Não há ativos separados de iluminação. A adaptação de cor de cada recorte é
calculada diretamente a partir da região do fundo onde a fruta será inserida.

Cada imagem recebe uma semente derivada da configuração, split e índice. A
geração pode ser interrompida e retomada sem alterar as amostras restantes.
`manifest.jsonl` registra fundo, mapa, recortes, semente e rejeições de cada
saída. Fundos e recortes de validação são disjuntos dos usados no treino.

## Geração confirmatória e desempenho

[`confirmatory_pool.yaml`](configs/synthesis/confirmatory_pool.yaml) gera o pool
aninhado de 1.040 imagens de treino e 260 de validação em 720×960, que é a
resolução efetivamente consumida antes do padding do treino com `imgsz=960`.
O gerador agrupa tarefas pelo fundo escolhido, mantém caches limitados por
processo para fundos, profundidade e frutas e envia aos workers somente
o identificador de cada cena. Sidecars regeneráveis usam rename atômico sem
forçar sincronização individual em disco; manifestos e resumos finais continuam
com escrita durável.

```bash
./run_pipeline.sh synthesize --workers 8 \
  --synthesis-config configs/synthesis/confirmatory_pool.yaml

# Medir 1, 4 e 8 workers com os ativos e o filesystem do servidor.
.venv/bin/python scripts/benchmark_synthesis.py \
  --asset-root data/assets/prepared \
  --workers 1 --workers 4 --workers 8 \
  --output artifacts/synthesis_benchmark.json
```

O benchmark não preserva as imagens temporárias. Compare a mediana de pelo menos
três repetições e escolha o número de workers pelo maior throughput estável, não
apenas pela quantidade de CPUs.

## Otimização autônoma

A skill pessoal `$tune-synthetic-fruit-detector` conduz buscas limitadas de
parâmetros do gerador e do detector, promove candidatos por etapas e pode
otimizar o código quando um perfil demonstra um gargalo. Cada execução exige
orçamento finito de trials, GPU-horas, paciência e melhoria mínima; todas as
tentativas são registradas em `artifacts/optimization/`.

O modo padrão `strict-synthetic` não consulta rótulos reais para selecionar
candidatos sintéticos. O modo `real-assisted` só pode ser ativado explicitamente
e muda a interpretação do resultado. Nenhum modo pode consultar o teste externo
antes do congelamento e da autorização para a avaliação final.

## Grade piloto

As condições disponíveis para validar a infraestrutura são:

- `real_baseline`: somente as 100 imagens reais de treino;
- `synthetic_depth`: somente as imagens sintéticas.

As duas condições usam a mesma validação real apenas para exercitar o fluxo. A
grade piloto compara YOLOv8n/YOLOv8s e não deve produzir os resultados do
estudo. Cada treino roda em um processo isolado e reutiliza resultados
completos; `last.pt` permite retomar uma execução interrompida.

Essa grade é um piloto operacional, não o desenho confirmatório do artigo. O
desenho confirmatório tem uma referência manual, uma condição controlada e
cinco condições somente sintéticas (`1x`, `2x`, `3x`, `5x` e `10x`). Não há
mistura de imagens reais e sintéticas no experimento principal.

O fluxo obrigatório é `train` → `select` → `test --unlock-test`.

## Execução em servidor

Para uma máquina única, execute dentro de `tmux` ou `screen`:

```bash
tmux new -s poncan
./run_pipeline.sh train --device 0 2>&1 | tee runs/server.log
```

Em Slurm, ajuste recursos/partição e envie
[`server/train.slurm`](server/train.slurm):

```bash
sbatch server/train.slurm
```

Use `PIPELINE_VENV=/caminho/.venv` para manter o ambiente fora do repositório e
`PYTHON_COMMAND=python3.11` para escolher o interpretador.

## Estrutura gerada

```text
data/
  raw/                  # fotos brutas de frutas e fundos
  assets/prepared/      # pacote público preparado
  assets/regenerated/   # DIS + ZoeDepth reconstruídos
  real_source/          # 130 originais normalizados apenas na orientação EXIF
  real_yolo/            # split materializado 100/15/15
  generated/            # datasets sintéticos e manifestos
artifacts/
  real_split.json
  model_selection.json
  test_results.json
runs/
  training/ validation/ test/
```

Dados, checkpoints e resultados locais são ignorados pelo Git. Não publique a
base real antes de confirmar a licença e o consentimento aplicáveis às imagens.

## Testes

```bash
./run_pipeline.sh help
.venv/bin/python -m pytest
```

`python setimages.py` funciona como um atalho para `depth_robust.yaml`, mas
grades experimentais devem usar os scripts acima.
