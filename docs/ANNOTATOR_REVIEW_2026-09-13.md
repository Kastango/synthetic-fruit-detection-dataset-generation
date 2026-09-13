# Revisão do anotador — 13/09/2026

Decisões fornecidas pelo usuário sobre os painéis em
`artifacts/annotator_review_cor_luz_2x/`. Referem-se ao detector da receita
original em synthetic-2x, semente 41, confiança 0,25. Não são classificações
visuais inferidas automaticamente.

## CitDet: alteração de escopo

Usar apenas `Fruit on Tree` (categoria COCO original 2), descartando
`Fruit on Ground` (1) das anotações de avaliação. Configuração permanente
em `configs/pipeline.yaml`, filtro antes de colapsar para classe 0.
119 imagens originais, 3.819 caixas de árvore; 6.263 caixas de chão excluídas.
Imagens são preservadas integralmente, inclusive eventuais negativos. Na
avaliação padrão, detecções sem correspondência com árvore contam como FP,
inclusive no chão. Não há máscara de exclusão espacial ou filtro de previsões
baseado no gabarito.

Os resultados anteriores com 10.082 caixas são históricos (árvore + chão)
e não são diretamente comparáveis ao novo protocolo. O manifesto muda;
avaliações anteriores não devem ser reutilizadas como se fossem tree-only.
O ZIP original e a cópia importada antiga são preservados.

## Manual_full_val: observações e implicações

| Caso | Observação do anotador | Implicação para investigar |
|---|---|---|
| M01 | Fruta entre galhos, cerca de 15% visível. | Oclusão severa; a receita atual exige visibilidade mínima de 30%. |
| M02 | Três frutos atrás da folhagem correspondem a três caixas maiores. | Preservar identidade de cada fruto mesmo quando sua superfície aparece fragmentada. |
| M03 | Anotação pequena demais; modelo tenta representar o fruto com duas caixas. | Há problema no gabarito e fragmentação da previsão; não atribuir tudo à receita. |
| M04 | Modelo acerta, mas a caixa é maior que o alvo. | Conferir convenção e extensão da caixa, não tratar como fruto ausente. |
| M05 | Modelo não enxerga frutas entre folhas com céu azul ao fundo. | Investigar oclusão e contexto de contraluz local. |
| M06 | Três previsões: duas para partes do fruto e uma de baixa confiança para o fruto inteiro. | Fragmentação/duplicação e confiança da caixa completa. |
| M07 | Fruta real não anotada, encontrada pelo modelo. | Falso positivo aparente decorrente de anotação ausente. |
| M08/M09 | Previsões sobre galhos e pontos verdes sem fruto. | Falsos positivos de fundo; candidatos a negativos difíceis. |

Nenhuma caixa manual foi alterada: a revisão identifica problemas, mas não
fornece coordenadas corrigidas. Próxima prioridade sugerida: sanity check de
frutos com oclusão severa e partes visíveis desconectadas, com caixa coerente
por instância, e negativos de galhos. Manter receita do usuário como base.
