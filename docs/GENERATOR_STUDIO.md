# Studio

## Instalar e abrir

Use Python 3.11 ou 3.12 na raiz do repositório.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python scripts/studio.py
```

Abra http://127.0.0.1:8765. Sem ativos, clique em "Baixar dados de demonstração".
O pacote contém seis fundos, seus mapas de profundidade e 32 recortes RGBA.
O clone inclui o ZIP de cerca de 7 MB. Se faltar, o servidor tenta baixá-lo.
O Studio verifica tamanho e SHA-256 antes de extrair em `data/studio-demo`.
A instalação não substitui pastas existentes.

O kit reduz a resolução e o catálogo para demonstração. Ele não reproduz
os dados dos experimentos. Os hashes das fontes e as transformações estão
no `provenance.json` do pacote. As fotografias pertencem à coleta do projeto;
a licença do código não concede licença independente sobre essas imagens.

Para usar seus próprios ativos preparados:

```bash
.venv/bin/python scripts/studio.py --asset-root /caminho/ativos
```

A pasta deve conter `backgrounds`, `backgrounds_map` e `pictures_trimmed`.
Cada fundo deve ter seu mapa correspondente, e os recortes devem preservar
a transparência. Sem esse argumento, o Studio prefere o catálogo completo
em `data/assets/regenerated` quando ele existe.

## Quantidade e variedade

Ajuste somente o mínimo e o máximo de frutas. A curva em U favorece
quantidades próximas dos extremos e mantém todos os valores possíveis.
Entre 10 e 100, a menor probabilidade fica no centro, em 55. Limites iguais
fixam a quantidade solicitada. Rejeições e oclusões podem reduzir as caixas.

O gerador sorteia um fundo por cena, com reposição. Fundos e frutas têm
50% de chance de espelhamento horizontal, com sorteios independentes.
O mapa de profundidade acompanha o fundo. A receita exporta
`augmentation.horizontal_flip: true`; use `false` no YAML para desativar.

A receita inicial está em `configs/synthesis/studio.yaml`. As receitas
antigas com `objects.dense` conservam a amostragem anterior.

## Sementes e reprodução

Mantenha a semente para comparar os ajustes nas mesmas cenas. A exportação
usa `sampling.mode: paired-v1`. Mudanças de aparência preservam os sorteios
de geometria. Alterar contagem, tamanho ou catálogo pode mudar as caixas.
Reproduzir pixels exige os mesmos ativos, código e bibliotecas.

"Salvar receita" baixa o YAML. "Gerar dataset" cria um ZIP com imagens,
caixas YOLO, receita, manifestos e hashes. O processamento usa CPU e salva
os resultados em `artifacts/studio`. Não exige dados reais anotados,
modelos de treino, DepthPro ou rembg para usar o kit preparado.

## Acesso na rede local

```bash
.venv/bin/python scripts/studio.py --host 0.0.0.0
```

Acesse `http://IP-DO-SERVIDOR:8765` em outra máquina da mesma rede.
