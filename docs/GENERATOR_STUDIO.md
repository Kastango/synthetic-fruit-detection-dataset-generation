# Estúdio e desenvolvimento do gerador

A meta é um único detector treinado com imagens sintéticas que funcione nas
duas coletas reais. Os ciclos de desenvolvimento usam **YOLOv8s**, 3×
(312 imagens de treino, 78 de validação), sementes de treino 41 e 42. A grade
com outros modelos e volumes fica para uma etapa posterior. A densidade
preserva a decisão do projeto: modo esparso 1–30, modo denso 60–110 em 6%
das cenas. Isso é uma mistura, não uma garantia de 2.093 caixas a cada 104 imagens.

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

A interface usa a receita atual. A candidata de pesquisa
`paired_essential.yaml`, que remove exposição independente e sombras
projetadas, piorou o detector nos dois cenários e não foi promovida à
interface. Os [resultados da comparação](RESULTS.md#primeira-comparação-pareada-com-yolov8s) estão registrados.
Os 13 sliders restringem os ajustes de uso cotidiano; constantes mecânicas
continuam existindo e aparecem no YAML.

## Sementes e reprodução

O novo modo é opt-in para configurações existentes:

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

## Experimento reduzido

O ciclo seguinte usa `configs/realism_yolov8.yaml`, com registro em
`artifacts/realism_yolov8`. O objetivo imediato é superar o YOLOv8s treinado
em `manual-full` no CitDet (mAP@.50:.95: 0,223432 na semente 41 e 0,204814
na 42; média 0,214123), preservando verossimilhança. A meta ampla continua
sendo aproximar também a qualidade local do treino real (média 0,549078).
Uma melhoria no CitDet não será descrita como cumprimento dessa meta ampla.

Cada hipótese deve registrar sua justificativa visual antes dos treinos,
comparar as sementes 41 e 42 com o mesmo protocolo e selecionar checkpoints
somente na validação sintética. Densidade, escala e aparência são examinadas
separadamente: uma média global de similaridade não pode esconder uma
distribuição implausível. A grade completa só se justifica após ganho nas
duas sementes do CitDet e inspeção visual favorável; a validação local deve
continuar reportada, inclusive quando houver regressão.

Hipótese inicial: a saturação nas caixas sintéticas está acima das duas
referências reais. A candidata `paired_saturation` aumenta de 0,35 para 0,70
a perda de saturação ligada à exposição, mantendo todo o restante. Os 390
arquivos de rótulos foram comparados byte a byte com `paired_reference`.
Não há aumento de frutas, novos controles ou seleção de cenas pelo detector.
Medidas em caixas incluem fundo e oclusão; a inspeção visual verifica se a
cor perde plausibilidade mesmo quando uma distância numérica melhora.

Segunda hipótese, definida antes dos resultados da primeira: remover o
achatamento adicional nas regiões claras (`bright_flatten_strength`) sobre
a candidata de saturação. O alvo hard-light já responde à cor local; a
mistura extra pode apagar relevo e produzir aparência plana. A ablação
`paired_relief` elimina uma folha do YAML, mantendo densidade e geometria.
As prévias e distribuições serão examinadas antes de autorizar seus treinos;
um ganho numérico sozinho não justifica promover a receita.

A primeira candidata terminou: `paired_saturation` obteve 0,213395/0,198733
no CitDet e 0,380718/0,382966 na validação local (sementes 41/42).
A distância de saturação diminuiu nas duas coletas, mas o mAP do CitDet
piorou em ambas as sementes contra `paired_reference`. Não foi promovida.
A segunda ablação segue como teste da contribuição do achatamento sobre
essa aparência, sem assumir que a correção de saturação seja uma melhoria
do detector.

Terceira hipótese: `paired_count_scale` parte novamente de `paired_reference`,
sem a alteração de saturação que piorou o CitDet. Somente as cenas com
contagem acima de `objects.max` recebem escala multiplicada por
`sqrt(objects.max / contagem_sorteada)`. A opção experimental é
`objects.dense.scale_with_count: true`; não acrescenta um limite numérico
nem muda a probabilidade de cenas densas. A relação entre área projetada e
contagem motiva a regra, mas ela não é uma lei física nem foi ajustada para
maximizar o detector. Deve ser conferida contra folhas, galhos e perspectiva.

Antes do treino, a correlação de Spearman entre contagem e mediana de tamanho
por imagem foi −0,448 no treino local, −0,183 no CitDet e −0,077 na referência
sintética. As cenas sintéticas acima de 60 caixas tinham mediana de tamanho
2,826% da imagem, contra 1,668% no CitDet. Essa diferença motiva testar frutas
menores nas cenas densas. As imagens esparsas devem permanecer idênticas;
os testes verificam reprodução entre workers com a nova regra ativada.

Quarta hipótese: a gradação da cena aplica contraste antes de brilho. O
contraste satura os claros em 255 e o brilho 0,82 multiplica o resultado já
ceifado, de modo que nenhum pixel sintético passa de 209. Medido em 60 fundos,
o percentil 99 do valor cai de 243,5 no fundo cru para 221 depois da gradação,
e a fração de pixels claros vai de 0,70% para 0,00%. As duas coletas reais
ficam em 243,0 (manual, 0,49%) e 240,0 (CitDet, 0,27%): os fundos crus já
correspondem ao real nessa característica e é a gradação que a elimina.

A candidata `paired_highlights` inverte a ordem para brilho, contraste e
saturação, sem mudar nenhum valor. Nas imagens geradas o brilho médio fica
praticamente igual (63,05 para 63,77) e a cauda volta: percentil 99 de 212
para 232, pixels claros de 0,00% para 0,13%. Os 390 arquivos de rótulos são
byte a byte iguais aos de `paired_reference`. A inspeção visual mostra o céu
entre as folhas voltando a estourar, como nas fotos reais, sem outra
alteração perceptível. Não há fruta nova, mudança de densidade nem de
geometria. Um teto uniforme de luminância é um indício global que o detector
pode aprender no lugar da aparência do fruto; a hipótese é que removê-lo
ajude nas duas coletas. A opção `output.scene_grading.exposure_first` deixa
o comportamento antigo como padrão, para preservar a receita de referência.

Quinta hipótese: a posição vertical dos frutos. Medida na fração de caixas
no terço superior da imagem, a referência sintética tem 35,9% contra 21,5%
no treino manual e 13,4% no CitDet; no terço inferior tem 20,7% contra 38,8%
do CitDet. O gerador sorteia qualquer ponto acima de `exclude_bottom_fraction`,
inclusive céu aberto, e as prévias mostram fruta contra o céu e sobre tronco
nu. A candidata `paired_canopy` ativa `placement.require_vegetation`, que já
existia: um indício cromático de vegetação por excesso de verde com limiar de
Otsu restringe os pontos de inserção. O indício não distingue grama de copa e
não certifica galho; o corte inferior continua evitando o chão.

Sem mudar densidade (17,8 contra 17,9 caixas por imagem), o percentil 10 da
posição vertical vai de 11,1 para 21,4 (manual real: 20,1) e o terço superior
cai para 20,9% (manual real: 21,5%). A distância de posição ao CitDet cai de
13,81 para 5,14. A saturação medida na caixa piora, de 7,90 para 16,03 contra
o manual, porque a caixa passa a conter folha verde em vez de céu cinza; isso
é consequência da colocação, não um ajuste de cor. Como a geometria muda, os
rótulos não são idênticos aos de `paired_reference` e a comparação é entre
receitas, não entre cenas pareadas.

Sexta hipótese: a faixa de exposição das cenas. O brilho médio por imagem
tem amplitude p5-p95 de 16,8 no sintético contra 32,6 no treino manual e 34,5
no CitDet. Os 228 fundos foram fotografados no mesmo tipo de luz difusa, então
todas as cenas sintéticas saem com a mesma exposição, enquanto uma coleta de
campo varia por hora do dia e ajuste de câmera. A candidata `paired_exposure`
acrescenta `output.scene_grading.brightness_jitter`, um fator por cena entre
0,85 e 1,45 aplicado ao brilho da gradação, sobre `paired_highlights`.

A gradação incide apenas no fundo, antes de colar fruta, de modo que o
`hsv_cast` leva o fruto junto com a luz da cena. O sorteio vem de um fluxo
próprio derivado da semente da cena, não do gerador de geometria: os 390
rótulos são byte a byte iguais aos de `paired_highlights`. O resultado tem
percentis 54,9/63,8/73,1/81,6/90,1 contra 57,6/64,9/71,1/79,5/91,8 do treino
manual real. A faixa foi escolhida por inspeção visual antes do treino, com
o limite superior no ponto em que a cena ainda parece um dia claro e não uma
imagem lavada; ela aproxima a coleta manual e não alcança o CitDet, que é
mais claro por ser outra coleta.

Sétima hipótese, derivada do modo de falha e não de uma distância de
distribuição. Na validação local a lacuna já vale 0,226 em IoU 0,50 e fica
plana até 0,80, então não é ajuste de caixa: é detecção. A precisão do modelo
sintético é 0,803 contra 0,925 do real, mas o recall é 0,595 contra 0,814.
Separando as 451 caixas da validação por saturação medida na caixa, o recall
sintético e real fica em 0,432/0,775 abaixo de 35, 0,488/0,869 entre 35 e 50,
0,659/0,906 entre 50 e 65 e 0,854/0,942 acima de 65. Na fruta viva o sintético
quase alcança o real; toda a lacuna está na fruta opaca, que o modelo real
encontra sem dificuldade. A fruta opaca não é intrinsecamente difícil, ela
falta no treino sintético.

O conjunto sintético tem 18,7% das caixas abaixo de saturação 50, contra 42,8%
no treino manual e 58,2% no CitDet. A causa está no modelo de aparência:
escurecer o canal V do HSV não altera a saturação, que é (max-min)/max e não
depende da escala, e a perda ligada à exposição usava `min(|fator-1|, 1)`,
valendo igualmente para sol e sombra. O efeito medido é uma inversão: a razão
entre a saturação da fruta no quartil escuro e no quartil claro é 0,76 no
treino manual e 0,97 no CitDet, mas 1,10 no sintético, isto é, sol lavado e
sombra vívida.

A candidata `paired_shade` trata isso como um mecanismo só, o modelo de sombra:
`desaturate_shade_only` restringe a perda ao lado abaixo de 1, e a faixa de
exposição e a intensidade são recalibradas para 0,20-1,30 e 1,0. A escolha veio
de uma varredura de quatro combinações em pools de 80 imagens, comparando
percentis contra as duas coletas antes de qualquer treino. O resultado leva a
fração de caixas opacas de 18,7% para 34,2% (manual real: 42,8%), o percentil
10 de 42,1 para 33,4 (27,7) e a razão escuro/claro de 1,10 para 0,85 (0,76).
Os 390 rótulos continuam byte a byte iguais aos de `paired_reference`. Na
inspeção visual a fruta em sombra profunda fica parda, como no interior da
copa, e a fruta iluminada mantém a cor. Uma correção anterior de saturação
(`paired_saturation`) moveu a marginal na direção certa e piorou o detector;
por isso a hipótese aqui não é a marginal, e sim a população ausente.

Oitava hipótese: a assinatura de frequência da imagem. As fotos reais são de
12 MP reduzidas cerca de 4,2 vezes para chegar a 960 px, enquanto a cena
sintética é composta nativamente em 720 x 960 e ainda recebe uma máscara de
nitidez (`sharpen_percent: 25`). Medindo a energia radial média do espectro,
normalizada pela banda baixa, o sintético tem 0,0899 na banda alta contra
0,0658 no treino manual e 0,0714 no CitDet, e 0,0501 na muito alta contra
0,0330 e 0,0408. A cena sintética é mensuravelmente mais crocante, e isso é
um indício global, presente em toda imagem, que o detector pode usar no lugar
da aparência do fruto.

A candidata `paired_sharpness` zera `sharpen_percent`, sem tocar em mais nada.
Em sondas de 80 imagens o espectro vai para 0,0755 na banda alta e 0,0419 na
muito alta, praticamente sobre o CitDet e bem mais perto do manual. Os 390
rótulos continuam byte a byte iguais aos de `paired_reference`: a máscara
incide no fundo, antes de colar fruta. A correção é por subtração; se a
receita for promovida, as três chaves de nitidez saem do YAML.

A sétima candidata terminou antes desta e não foi promovida. `paired_shade`
obteve 0,1869/0,1928 no CitDet e 0,3607/0,3734 na validação local, a maior
perda medida no ciclo. O diagnóstico posterior mostra por que a hipótese
estava errada: o recall na faixa de saturação abaixo de 35 ficou em 0,405
contra 0,432 da referência, ou seja, fruta sintética opaca não ensinou o
detector a encontrar fruta real opaca, e ainda custou recall na fruta viva
(0,801 contra 0,854). A fração de pixels alaranjados dentro da caixa já
coincidia entre sintético e real antes da mudança (31,8% das caixas abaixo
de 20%, contra 33,7% no manual e 33,6% no CitDet), então a população
ausente não era de fruta opaca e sim, provavelmente, de fruta vista atrás
de folhagem, que a dessaturação não reproduz.

Nona hipótese: a estrutura da oclusão. O conjunto sintético já contém oclusão
em quantidade: comparando o mesmo pool anotado em `visible` e em `amodal`,
27,2% das instâncias ficam abaixo de 70% de visibilidade e 14,2% abaixo de 50%.
Mesmo assim o recall nas caixas com menos de 10% de pixels alaranjados é 0,375,
contra 0,910 nas caixas acima de 45%. A suspeita é a forma da oclusão, não a
quantidade: `occlusion.depth_smooth_radius` borra o mapa de profundidade antes
do limiar e apaga oclusores finos, deixando manchas grossas. A candidata
`paired_occlusion` zera esse raio. Em sondas de 80 imagens, 55,2% das caixas
mudam de área, com razão mediana 1,042 e percentil 90 em 1,586, então a mudança
é estrutural e não cosmética. O resultado foi 0,1991 no CitDet e 0,3702 na
validação local, cerca de 2,8 desvios abaixo da referência no CitDet.

Décima hipótese: a fração de fruta verde. A augmentação de treino usa
`hsv_h: 0.015`, ou seja, quase não desloca o matiz, ao contrário de saturação e
valor. Diferenças de matiz são portanto visíveis para o detector. Medindo o
matiz mediano do núcleo interno de cada caixa, o sintético tem 17,5% de fruta
verde contra 6,0% no treino manual, 5,3% na validação local e 13,7% no CitDet.
A precisão do modelo sintético é 0,803 contra 0,925 do real, compatível com um
detector treinado com excesso de esfera verde. `paired_ripeness` baixa
`ripeness.fraction_affected` de 0,18 para 0,06. A sonda mostra que o parâmetro
responde por cerca de quatro pontos e que há um piso de aproximadamente 13%
vindo dos próprios recortes do catálogo, alguns naturalmente verdes: o
resultado é 14,0% de verde, sobre os 13,7% do CitDet. O treino ficou em 0,2037
no CitDet e 0,3775 na validação local, ou seja, perda de 0,0067 e empate.

Décima primeira hipótese: o piso de visibilidade. `placement.min_visibility`
rejeita colocações abaixo de 15% de área visível, e o diagnóstico mostra que é
justamente a fruta muito escondida que o detector não encontra: recall de 0,375
nas caixas com menos de 10% de pixels alaranjados, contra 0,910 acima de 45%.
`paired_visibility` baixa o piso para 0,05. A contagem de caixas por imagem não
muda, 18,8 nos dois casos, porque uma colocação recusada é sorteada de novo; o
que muda é a mistura, com as caixas abaixo de 20% de laranja indo de 31,2% para
34,7%, contra 33,7% do treino manual. O resultado foi 0,1968 no CitDet e 0,3679
na validação local, cerca de 3,4 desvios abaixo da referência.

Décima segunda: mistura em vez de deslocamento. As receitas de aparência
produzem rótulos byte a byte iguais, isto é, as mesmas cenas renderizadas de
outra forma, o que permite montar um conjunto misto sem código novo.
`paired_mixture` alterna cena a cena entre `paired_reference` e
`paired_sharpness`, 195 imagens de cada, aumentando a variância no eixo de alta
frequência sem mover o centro. É o único candidato que não perdeu no CitDet:
0,2105 contra 0,2104 da referência, enquanto a receita pura de nitidez havia
perdido 0,0096. Na validação local, porém, caiu para 0,3667. Uma mistura
recupera o eixo externo e custa no local; não é um ganho conjunto.

Antes de aceitar o padrão de doze resultados negativos, a receita de referência
foi regerada com o código atual e comparada byte a byte com o pool que foi
treinado: as 312 imagens de treino são idênticas e o `config_hash` coincide,
apesar de o SHA-256 do compositor ter mudado. As opções acrescentadas são
neutras na saída quando não são ativadas, e as comparações do ciclo são válidas.

A seleção de checkpoint também foi verificada, porque ela usa a validação
sintética. Comparando `best.pt` e `last.pt` nas quatro corridas da referência,
`best.pt` vence por 0,007 a 0,018 em três delas e empata na quarta, nos dois
cenários. A regra de seleção atual não está deixando desempenho na mesa.

## Controle de reprodutibilidade e leitura do ciclo

Depois de oito candidatas todas abaixo da referência no CitDet, a mesma receita
`paired_reference` foi treinada com duas sementes novas, 43 e 44, em
`configs/replicate_yolov8.yaml` e `runs/replicate_yolov8`. Os valores no CitDet
são 0,214497 e 0,205714 nas sementes 41 e 42, e 0,215649 e 0,205761 nas 43 e 44:
médias por par de 0,210106 e 0,210705, com desvio de 0,0054 entre as quatro
corridas. Na validação local as médias por par são 0,378725 e 0,377744, com
desvio de 0,0103.

O desvio da média de duas sementes é portanto cerca de 0,004 no CitDet e 0,007
na validação local. Uma estimativa anterior, obtida juntando a diferença entre
sementes de receitas diferentes, chegava ao dobro disso: aquele número mistura
a diferença entre receitas com o ruído dentro de uma receita e não serve como
piso. Com o controle correto, `paired_canopy`, `paired_sharpness`,
`paired_relief` e `paired_shade` ficam entre duas e cinco vezes o desvio abaixo
da referência, ou seja, são pioras medidas e não sorteio; `paired_count_scale`
e `paired_highlights` empatam. Nenhuma candidata superou a referência em
nenhum dos dois cenários.

Com quatro sementes, a referência sintética marca 0,2104 no CitDet contra
0,2141 do YOLOv8s treinado em `manual-full`, uma diferença de 0,0037, da ordem
de um desvio. No CitDet os dois não se distinguem. A lacuna que permanece é a
da validação local, 0,378 contra 0,549.

O ciclo também explica por que quase tudo ficou plano. O protocolo de treino
usa `hsv_s: 0.7` e `hsv_v: 0.4`, isto é, a cada época a augmentação já sorteia
saturação e valor de cada imagem em faixas maiores do que as diferenças que as
candidatas de cor introduziam. A calibração de exposição de `paired_exposure`,
0,85 a 1,45, cabe inteira dentro de `hsv_v`. As candidatas que mexeram em cor,
exposição e destaques eram invisíveis para o detector por construção; as que
produziram efeito acima do ruído foram justamente as que a augmentação não
desfaz, ou seja, geometria (`paired_canopy`), conteúdo de alta frequência
(`paired_sharpness`) e estrutura de contraste do fruto (`paired_relief`,
`paired_shade`), e as quatro pioraram. Hipóteses futuras nessa configuração
devem começar por aí, e não por distâncias de distribuição de cor.

A interação com a augmentação foi testada diretamente em
`configs/lowaug_yolov8.yaml`, com a mesma receita de referência e apenas
`hsv_s` de 0,7 para 0,3 e `hsv_v` de 0,4 para 0,2. O resultado piorou nos dois
cenários: 0,1988 no CitDet e 0,3713 na validação local, contra 0,2104 e 0,3782
da referência. A augmentação de cor não está encobrindo o realismo do gerador,
ela está cobrindo uma lacuna que o conjunto sintético não cobre sozinho. É por
isso que ajustar cor na geração é redundante, e não porque o efeito se perde.
Essa é uma sonda de protocolo, não uma receita: o número não é comparável ao
treino real, que usa a augmentação completa.

## Situação ao fim do ciclo

Doze candidatas de gerador e uma sonda de protocolo foram medidas contra
`paired_reference`, todas com duas sementes e avaliação nos dois cenários.
Nenhuma superou a referência em nenhum dos dois. Sob o piso de ruído do
controle, `paired_count_scale` e `paired_highlights` empatam, `paired_saturation`
e `paired_exposure` ficam perto de um desvio, e `paired_canopy`,
`paired_sharpness`, `paired_occlusion`, `paired_relief` e `paired_shade` são
pioras de duas a cinco vezes o desvio. `paired_ripeness` e `paired_visibility` também
regridem, a primeira dentro de dois desvios e a segunda em 3,4. A receita de
referência permanece a recomendada e nada foi promovido. A grade completa não
foi executada porque ela só se justifica após um ganho, e não houve.

Doze perturbações de mecanismo único a partir da mesma receita, todas negativas
ou empatadas, com média de cerca de 0,009 abaixo no CitDet, descrevem um ótimo
estreito e não um platô. Isso é coerente com a origem da receita, que já é
produto de afinação em ciclos anteriores. As duas alavancas que a história do
projeto registra como eficazes, a exposição por instância e a densidade de
cena, já estão nela nos valores escolhidos.

O que resta como caminho não é outro parâmetro de aparência. As distribuições
mensuráveis já coincidem, a augmentação cobre cor e luminância, a seleção de
checkpoint está correta e as mudanças de estrutura pioram. A única alavanca que
a história do projeto associa a um salto no CitDet é a proporção de cenas
densas, hoje fixada em 6% por uma decisão de projeto que casa as 104 imagens
com as 2.093 caixas reais do conjunto manual. Alterá-la é uma decisão de
escopo, não uma hipótese técnica, e não foi tomada aqui.

As distribuições que este ciclo sabe medir já coincidem entre sintético e real:
contagem por imagem, tamanho da caixa no espaço de entrada da rede, posição
vertical, contraste entre caixa e entorno, fração de pixels alaranjados dentro
da caixa e proporção de instâncias parcialmente ocluídas. O que resta de
diferença não aparece nesses descritores.

O diagnóstico por recall delimita onde procurar em seguida. Na validação local
a lacuna já vale 0,226 em IoU 0,50 e permanece plana até 0,80, então não é
ajuste de caixa. Separando por fração de pixels alaranjados, o recall sintético
é 0,910 acima de 45% e 0,375 abaixo de 10%. O detector sintético praticamente
alcança o real na fruta bem visível e falha na fruta vista atrás de folhagem.
Duas tentativas de atacar essa população, por dessaturação (`paired_shade`) e
por oclusão mais fina (`paired_occlusion`), pioraram; a aparência local do
encontro entre folha e fruto continua sem hipótese testada que a descreva.

```bash
.venv/bin/python scripts/generate_synthetic.py \
  --synthesis-config configs/synthesis/paired_reference.yaml --workers 6
.venv/bin/python scripts/generate_synthetic.py \
  --synthesis-config configs/synthesis/paired_essential.yaml --workers 6
.venv/bin/python scripts/train_grid.py \
  --config configs/similarity_yolov8.yaml --device 0 --workers 4
```

Depois dos quatro treinos, avalie os dois cenários:

```bash
.venv/bin/python scripts/evaluate_similarity.py --device 0
```

São quatro treinos: duas receitas × duas sementes × **somente YOLOv8s**.
Os diretórios `runs/similarity_yolov8` e `artifacts/similarity_yolov8` isolam
essa rodada dos pesos e relatórios históricos. A receita de referência
precisa de um novo controle porque a amostragem pareada usa outro pool.

A receita só deve ser promovida se a inspeção visual e as distribuições
sustentarem a mudança e o mAP@.50:.95 não piorar em nenhum dos dois cenários,
com melhoria observada em pelo menos um. Devem ser registradas as duas sementes,
sem escolher apenas o melhor resultado. Isso é um critério de desenvolvimento,
não um teste de significância. Checkpoints continuam selecionados na validação
sintética; as avaliações reais são exploratórias. Uma confirmação exige nova
coleta reservada e receita congelada antes de acessá-la.

## Comparação com dados reais na pesquisa

Esta análise não faz parte da interface e não é necessária para gerar dados:

```bash
.venv/bin/python scripts/compare_similarity.py \
  --dataset data/generated/paired_reference \
  --output artifacts/studio/reference_similarity.json
```

O script compara todo o split sintético com o treino local (104 imagens)
e o CitDet já usado no desenvolvimento (119 imagens). Mede quantidade,
escala, posição, luminância, saturação, contraste e pixels claros saturados.
A distância é a média da diferença absoluta entre 101 quantis, nas unidades
da característica. Não há uma pontuação geral de realismo.

Aparência dentro de caixas inclui folhagem e oclusões. Essas medidas não
identificam toda colagem implausível, fruta solta ou contexto inadequado;
a inspeção visual continua necessária. Os resultados de detector e a
similaridade são reportados separadamente, por cenário.

## Decisões preservadas do histórico

A revisão da thread `d1776008-a031-4593-ae55-e62233f741a2`, até 7 de setembro
de 2026, orienta este ciclo. Relatos de experimentos removidos são evidência
histórica, sem revalidação independente dos pesos. A grade multiarquitetura
que estava em execução foi interrompida para priorizar YOLOv8s, preservando
os checkpoints. Mantêm-se os 127 recortes elegíveis e a densidade escolhida.

O melhor sintético histórico do YOLOv8s alcançou 0,240 no CitDet, mas ficou
abaixo do treino real na validação local. A meta conjunta continua aberta.
Forma, manchas e desfoque não tiveram melhora simultânea convincente no
histórico; isso não justifica recolocar todos esses controles nem atribuir
o platô somente à quantidade de recortes. As 2.093 caixas reais pertencem
às 130 imagens: são 1.642 no treino e 451 na validação.

Antes de `paired-v1`, o hash da configuração alterava as sementes e sorteios
de aparência consumiam o fluxo da geometria. Uma pequena ablação podia mudar
fundos, posições e oclusões. A separação atual permite comparar aparência
com as mesmas caixas; testes verificam reprodução entre workers.

As próximas hipóteses devem partir de defeitos visíveis: bordas e luz dos
recortes, contexto dos fundos e pontos de inserção, ou relação entre escala
e densidade. Alguns fundos apresentam folhas menores e textura mais densa
que as referências de cítricos; uma cor semelhante não resolve isso.
Compare um mecanismo por vez, com cenas pareadas e duas sementes, e exija
melhora conjunta. A comparação já concluída está em [RESULTS.md](RESULTS.md#primeira-comparação-pareada-com-yolov8s).

## Direção visual

Árvores preenchidas com poncãs ocupam o centro; os 13 controles ficam à
esquerda e se reorganizam no celular. Fotos mantêm a proporção. Fundo,
caixas e detalhes são modos acionados pelo usuário, com estatísticas
recolhidas e sem comparação com dados reais na interface.
A paleta usa papel `#F6F8F7`, branco `#FFFFFF`, texto `#233A33`, folhagem
`#346851`, fruto `#EF9A35` e divisões `#D6DFDA`. A tipografia é Pomar Sans,
subconjunto local de Lato com licença incluída. Textos e títulos se limitam
ao necessário para ajustar, salvar e gerar dados.
