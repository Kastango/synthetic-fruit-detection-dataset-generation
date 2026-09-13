# Resultados e validação

A grade de 42 treinos está interrompida para revisão do gabarito e da receita.
Não há resultados de detectores publicados para a versão atual. As rodadas
exploratórias, a grade aposentada e suas tabelas foram removidas a pedido do
usuário. O histórico Git permanece intacto.

A receita aprovada pelo usuário é o padrão do Studio: 1% de negativos,
1–60 objetos solicitados com distribuição beta, centro de escala por cena
0,018–0,15, dispersão 2,24, visibilidade mínima 15% e piso de 60 pixels.
O pool anterior precisa ser regenerado. O protocolo está em [DATASETS.md](DATASETS.md).

`oranges_field` é uma coleta externa usada para desenvolvimento: seu protocolo
e suas estatísticas influenciaram a receita. Uma avaliação confirmatória requer
outra coleta intocada. `manual_full_val` também participa da seleção dos
checkpoints de `manual-full`.

## Auditar manual-full

A validação automática verifica rótulos, contagens, IDs congelados e hashes de
imagens/anotações contra o manifesto importado:

```bash
.venv/bin/python scripts/validate_data.py --stage real
```

A primeira passou para as 130 imagens e 2.093 caixas. O perfil também decodificou
todas as imagens: nenhum arquivo inválido, rótulo ausente, repetição exata de
imagem ou caixa duplicada na base manual. Isso não certifica que todas as
frutas estejam anotadas nem que cada caixa esteja visualmente correta.

Para revisar todas as imagens, sem amostragem e sem exigir checkpoint:

```bash
.venv/bin/python scripts/audit_dataset.py \
  --images data/real_yolo_confirmatory/images/train \
  --labels data/real_yolo_confirmatory/labels/train \
  --output artifacts/auditoria/manual_full_train.jsonl --port 8770

.venv/bin/python scripts/audit_dataset.py \
  --images data/real_yolo_confirmatory/images/val \
  --labels data/real_yolo_confirmatory/labels/val \
  --output artifacts/auditoria/manual_full_val.jsonl --port 8771
```

Abra [treino: 104 imagens](http://127.0.0.1:8770) e
[validação: 26 imagens](http://127.0.0.1:8771). Teclas 1–6 marcam `ok`,
`faltando`, `caixa-frouxa`, `duplicada`, `oclusao-extrema` e `nao-e-fruta`;
Enter salva e avança. As filas já estão em execução na sessão local.
Se o navegador estiver em outra máquina, use encaminhamento dessas portas.

```bash
.venv/bin/python scripts/audit_dataset.py \
  --output artifacts/auditoria/manual_full_train.jsonl --resumo
.venv/bin/python scripts/audit_dataset.py \
  --output artifacts/auditoria/manual_full_val.jsonl --resumo
```

A auditoria humana ainda depende dos vereditos do anotador. A ferramenta
registra julgamentos e não muda coordenadas. A [revisão anterior do
anotador](ANNOTATOR_REVIEW_2026-09-13.md) foi preservada como evidência útil.

## Retomar a pesquisa

O protocolo usa 7 condições, 3 modelos e sementes 41/42, sem congelamento.
YOLOv8s/YOLO26s usam SGD e batch 8; RT-DETR-L usa AdamW e batch 2.
Confira os demais parâmetros em [confirmatory.yaml](../configs/confirmatory.yaml).

O pool anterior foi criado com outra receita e outro código. Seus hashes
foram preservados. Regere o pool e seus subconjuntos antes de retomar a grade.
A revisão visual do manual-full continua pendente dos vereditos do anotador.
