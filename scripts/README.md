# Scripts

Execute os scripts na raiz do repositório, com o ambiente de `.venv`. A lógica
reutilizável fica em `fruit_pipeline/`; cada script trata os argumentos e chama
o pacote.

## Pipeline

`run_pipeline.sh` chama [`reproduce.py`](reproduce.py), que executa os scripts
abaixo nesta ordem. As etapas `prepare` e `all` percorrem a sequência;
`./run_pipeline.sh help` lista as etapas que podem ser executadas isoladamente.

| Script | Função |
|---|---|
| [`download_data.py`](download_data.py) | Baixa as fotos de frutas e os fundos. |
| [`download_real.py`](download_real.py) | Baixa o ZIP da base real anotada. |
| [`import_real_dataset.py`](import_real_dataset.py) | Importa a base real e confere 130 imagens e 2.130 caixas. |
| [`split_real.py`](split_real.py) | Congela a divisão real em 104 e 26 imagens. |
| [`preprocess_assets.py`](preprocess_assets.py) | Gera os recortes IS-Net e os mapas DepthPro. |
| [`materialize_controlled.py`](materialize_controlled.py) | Monta a condição `controlled`. |
| [`catalog_assets.py`](catalog_assets.py) | Cataloga fundos, mapas e recortes. |
| [`generate_synthetic.py`](generate_synthetic.py) | Gera o pool sintético. |
| [`materialize_nx_subsets.py`](materialize_nx_subsets.py) | Recorta os subconjuntos aninhados de 1x a 10x. |
| [`validate_data.py`](validate_data.py) | Audita dados, divisões e rótulos antes do treino. |
| [`train_grid.py`](train_grid.py) | Treina a grade; [`train_one.py`](train_one.py) roda cada execução em processo próprio. |
| [`select_models.py`](select_models.py) | Escolhe os checkpoints pela validação de origem. |
| [`download_external.py`](download_external.py) | Obtém e valida o pacote do conjunto externo. |
| [`import_external_test.py`](import_external_test.py) | Importa um conjunto de avaliação e grava seu manifesto. |
| [`evaluate_test.py`](evaluate_test.py) | Avalia os checkpoints escolhidos; [`evaluate_one.py`](evaluate_one.py) avalia cada um. |
| [`generate_report.py`](generate_report.py) | Consolida o relatório e exporta os CSVs de análise. |

[`curate_oranges_field.py`](curate_oranges_field.py) seleciona os recortes da
coleta externa antes da importação.
[`download_weights.py`](download_weights.py) baixa os 42 checkpoints publicados,
que dispensam `train` e `select`.
[`pack_checkpoints.py`](pack_checkpoints.py) monta esse pacote para a release.

## Figuras e exemplos dos resultados

Os comandos e a ordem de execução estão em
[Reproduzir os resultados](../docs/RESULTS.md#reproduzir-os-resultados).

| Script | Produz |
|---|---|
| [`plot_confirmatory_results.py`](plot_confirmatory_results.py) | Rankings, curvas por volume sintético e tabelas intermediárias. |
| [`plot_grid_diagnostics.py`](plot_grid_diagnostics.py) | AP por limiar de IoU, contagem prevista contra real e progressão por época. |
| [`find_detection_gap.py`](find_detection_gap.py) | Seleção da imagem com maior diferença de acertos na validação própria. |
| [`render_detection_examples.py`](render_detection_examples.py) | Detecções de cada condição sobre uma mesma imagem. |
| [`render_synthetic_examples.py`](render_synthetic_examples.py) | Cenas sintéticas com o gabarito desenhado. |
| [`build_example_sheets.py`](build_example_sheets.py) | Folhas que reúnem os exemplos. |
| [`check_documentation.py`](check_documentation.py) | Conferência de links e números publicados. |

Os geradores dos fluxogramas ficam em [`diagrams/`](diagrams) e estão descritos
no [guia das figuras](../docs/DIAGRAMS.md).

## Ferramentas interativas

- [`studio.py`](studio.py) abre o Studio de geração, descrito no
  [guia do Studio](../docs/GENERATOR_STUDIO.md).
- [`audit_dataset.py`](audit_dataset.py) abre a revisão visual do gabarito,
  descrita em [Auditar as anotações](../docs/RESULTS.md#auditar-as-anotações).

## Análises exploratórias

Estes scripts orientaram o desenvolvimento do gerador. Os resultados publicados
não dependem deles.

- [`analyze_misses.py`](analyze_misses.py) perfila as frutas que um checkpoint
  deixa de detectar.
- [`evaluate_operating_points.py`](evaluate_operating_points.py) calcula
  precision e recall com correspondência um a um em limiares fixos.
- [`measure_dataset_profiles.py`](measure_dataset_profiles.py) mede caixas,
  negativos e aparência dos conjuntos.
