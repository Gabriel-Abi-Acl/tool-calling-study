# tool-calling-study

Estudo comparativo de estratégias de uso de ferramentas em um loop de agente.
O recorte é metodológico: as mesmas tarefas sintéticas, três formas de pedir e ler uma chamada, e métricas definidas antes de qualquer corrida com modelo.

O repositório tem as interfaces, as ferramentas falsas, o loop, o dataset versionado e o harness offline. Os números do estudo entram no PR4, depois da medição com modelo real.

## Objetivo

Comparar três estratégias de tool use sobre a mesma interface de LLM:

1. ReAct: o modelo escreve um traço com `Action` e `Action Input`.
2. Function calling nativo: o modelo devolve a chamada no formato estruturado do provedor.
3. Plano JSON: o modelo lista os passos; o loop executa a lista e só então pede a resposta final.

As tarefas cobrem três situações: uma ferramenta, duas ferramentas em sequência, e abstenção (nenhuma ferramenta deve ser chamada). A ferramenta `weather` é uma distração fictícia: fica exposta no catálogo e não entra na sequência esperada do dataset v1.

O harness grava, para cada tarefa e estratégia:

- sequência chamada (nome e argumentos)
- se essa sequência de nomes casa com o gabarito
- se os argumentos estão corretos
- chamadas desnecessárias, inclusive `weather` quando o gabarito não pede clima
- se o agente se absteve
- resposta final
- latência
- tokens e custo estimado, quando o cliente informa

Este README não publica acurácia, qualidade nem custo de modelo.

## Hipótese

A hipótese, ainda sem medição, é que o function calling nativo preserva melhor a forma dos argumentos do que o traço ReAct em texto livre, e que o plano JSON reduz chamadas desnecessárias nas tarefas de abstenção porque os passos ficam explícitos antes da execução. ReAct pode perder uma chamada por erro de formato. O plano JSON pode inserir um passo que a tarefa não pede. O PR4 mede essas diferenças.

## Metodologia

- Cliente de LLM independente do provedor. A suíte usa um cliente falso com respostas roteirizadas. Sem a próxima resposta no roteiro, o cliente falha em vez de completar o texto.
- Três estratégias sobre esse cliente: `ReactToolStrategy`, `NativeToolStrategy` e `JsonPlanStrategy`. ReAct e o plano JSON pedem texto. A estratégia nativa envia o schema das ferramentas.
- Ferramentas falsas com esquema tipado: `lookup` (base fixa), `calculator` (aritmética por AST, sem `eval`) e `weather` (previsão fictícia). Consulta desconhecida devolve `not_found`.
- Fixture curta de 11 tarefas em `src/tool_calling_study/tasks/fixtures.json`, usada pelo loop e pelos testes do scaffold. O gabarito fechado do estudo é o dataset v1, descrito abaixo.
- O loop executa a estratégia, chama as ferramentas falsas e registra ferramenta, argumentos, resposta final e chamadas desnecessárias. Chamada desnecessária, neste registro do loop, é ferramenta fora do gabarito ou listada como proibida. Repetir uma ferramenta esperada não entra nessa lista.
- O harness conta a repetição como `extra_call`. O comando abaixo percorre o dataset v1. Com `LLM_PROVIDER=fake` e sem roteiro, cada tarefa termina em falha de transporte e `correct` fica nulo. Isso não é um resultado do estudo.
- No dataset v1, `reference_answer` é a saída da última ferramenta quando a tarefa pede chamada. Na abstenção com texto exato, é essa frase. Quando o agente deve recusar, o campo é `null`.

Os enunciados das tarefas estão em português. Os protocolos (`Action`, `Action Input`, `Final Answer`, o objeto de plano e os schemas) estão em inglês, no formato usado por essas estratégias.

Um turno ReAct válido tem uma `Action` e um `Action Input` em JSON, ou um `Final Answer`. O plano válido é um objeto `{"steps": [...], "final_answer": "..."}`. Argumentos que não são objeto JSON geram erro de parse, registrado pelo loop como `stopped_reason=parse_error`.

## Dataset de avaliação

O gabarito fechado está em `data/eval/tasks.v1.jsonl`: uma tarefa JSON por linha. O arquivo reúne 45 tarefas sintéticas, 15 em cada categoria:

- `single`: uma chamada, `lookup` ou `calculator`.
- `multi`: duas chamadas em cadeia. A segunda leva a saída da primeira em algum argumento.
- `abstain`: nenhuma chamada. O catálogo continua visível. Parte dessas tarefas tem frase exata em `reference_answer`; nas outras o campo é `null`, porque a resposta correta é recusar.

Cada linha tem:

- `id`
- `category`
- `prompt`: enunciado em português.
- `allowed_tools`: ferramentas expostas ao agente. O catálogo do v1 é sempre `lookup`, `calculator` e `weather`. Não é a lista do gabarito.
- `tool_call_allowed`: `false` na abstenção, mesmo com ferramentas expostas.
- `expected_sequence`: nome da ferramenta e `argument_constraints`. No v1, cada restrição é o valor exato do argumento.
- `reference_answer`: string ou `null`.

`weather` entra em `allowed_tools` e não entra em `expected_sequence`. Populações e previsões são constantes redondas deste estudo, não censo nem clima real. Os enunciados e as constantes são sintéticos e públicos.

`load_eval_tasks` lê o arquivo, confere o schema e recusa linha malformada: campo ausente ou inesperado, ferramenta fora do catálogo falso, argumento que o schema não aceita, cadeia que ignora a saída anterior, ou `reference_answer` diferente da última ferramenta. Para inspecionar:

```bash
uv run python -c "from collections import Counter; from tool_calling_study.tasks import load_eval_tasks; tasks = load_eval_tasks(); print(len(tasks)); print(dict(Counter(task.category for task in tasks)))"
```

## Como rodar

Requisitos: Python 3.11+ e [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

A configuração é por variável de ambiente. `.env.example` lista os nomes; os valores de chave ficam vazios no git. Para um experimento local posterior, exporte as variáveis no ambiente. O arquivo `.env` está no `.gitignore`.

`LLM_PROVIDER=fake` é o padrão. `openai` e `anthropic` importam o SDK correspondente só dentro da primeira chamada ao vivo. Sem a chave, a chamada para antes do import. Sem o pacote do SDK instalado no ambiente, a chamada levanta `ProviderNotInstalled` e não abre rede. Esses SDKs não fazem parte das dependências de desenvolvimento deste repositório.

O CI usa GitHub Actions, instala uv pelo release `astral-sh/setup-uv@v10.3.3` e roda ruff e pytest.

## Harness de avaliação

O comando percorre o dataset v1 com as ferramentas falsas e o cliente configurado. O padrão é o cliente falso, sem rede e sem chave. Provedor `openai` ou `anthropic` fica bloqueado até alguém passar `--allow-live`. A suíte não usa essa flag. Os experimentos ao vivo continuam no PR4.

```bash
uv run python -m tool_calling_study.evaluate --strategy all
uv run evaluate-tools --strategy react
uv run evaluate-tools --strategy json_plan --output /tmp/eval-json-plan.jsonl
```

`--strategy` aceita `react`, `native`, `json_plan` ou `all`. `--dataset` troca o JSONL; o padrão é `data/eval/tasks.v1.jsonl`. A saída é JSONL, um objeto por tarefa. Com `--strategy all`, a ordem é `react`, depois `native`, depois `json_plan`, e dentro de cada estratégia a ordem do arquivo. Sem `--output`, o JSONL vai para stdout. O stderr só informa a contagem de registros, sem taxa de acerto.

Cada objeto tem:

- `task_id`, `strategy`, `category`
- `selected_sequence`: lista de `{"name", "arguments"}` na ordem executada
- `sequence_match`: os nomes, na ordem, batem com `expected_sequence`
- `arguments_correct`: a sequência de nomes bate e cada argumento é igual à restrição
- `unnecessary_calls`: nome fora da posição esperada, ou chamada a mais (inclui repetir a ferramenta esperada)
- `abstained`: nenhuma ferramenta foi chamada
- `final_answer`, `answer_correct`
- `correct`
- `failure_modes`
- `latency_ms`
- `prompt_tokens`, `completion_tokens`, `estimated_cost_usd`
- `stopped_reason`: `final`, `parse_error`, `max_turns` ou `transport_error`
- `error`: mensagem curta no transporte; `null` nos outros casos

`answer_correct` é `null` quando `reference_answer` é `null`, porque não há frase para comparar. A comparação de texto, quando existe, é exata.

Modos de falha, só quando a corrida foi julgada:

- `wrong_tool`: o nome naquela posição não é o do gabarito, ou uma chamada extra usa ferramenta que o gabarito não tem
- `bad_arguments`: o nome bate e os argumentos não
- `extra_call`: houve mais chamadas do que o gabarito
- `missed_call`: faltou chamada no fim da sequência
- `retrieved_distractor`: `weather` foi chamada e o gabarito não pede `weather`
- `failed_abstention`: a tarefa é de abstenção e alguma ferramenta foi chamada

Abstenção: `correct` só é verdadeiro se nenhuma ferramenta foi chamada. Se `reference_answer` é texto, a resposta final também precisa ser esse texto. Se `reference_answer` é `null`, basta não chamar ferramenta. Erro de parse sem chamada, nesse caso, conta como não ter chamado. Erro de transporte deixa `correct`, `sequence_match`, `arguments_correct`, `answer_correct` e `abstained` nulos, e `failure_modes` vazio. Chamadas já executadas continuam em `selected_sequence`. O harness não transforma rede, roteiro esgotado ou chave ausente em erro do agente.

O `FakeLLM` devolve a resposta do roteiro como está. Sem uso na resposta, `prompt_tokens`, `completion_tokens` e `estimated_cost_usd` saem nulos. O harness soma cada campo só quando todas as respostas da tarefa o trouxeram. Não há tabela de preço no repositório. Os adaptadores ao vivo ainda não preenchem uso; isso fica para o PR4.

O cliente falso sem roteiro esgota na primeira chamada. O JSONL dessa corrida tem `correct` nulo de ponta a ponta. Não cite esse arquivo como acurácia do estudo.

## Roadmap

- PR4: experimentos com modelos reais e relatório.

## Layout

- `data/eval/tasks.v1.jsonl`: dataset versionado e o gabarito fechado.
- `src/tool_calling_study/llm`: interface, cliente falso e provedores com import tardio.
- `src/tool_calling_study/strategies`: parsers e as três estratégias.
- `src/tool_calling_study/tools`: esquemas e ferramentas falsas.
- `src/tool_calling_study/tasks`: fixture curta do loop e `load_eval_tasks`.
- `src/tool_calling_study/agent`: loop que grava o rastro.
- `src/tool_calling_study/evaluate.py`: harness e métricas offline.
- `src/tool_calling_study/checks.py`: detecção de chamada desnecessária no loop.
