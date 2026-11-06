# tool-calling-study

Estudo comparativo de estratégias de uso de ferramentas em um loop de agente.
O recorte é metodológico: as mesmas tarefas sintéticas, três formas de pedir e ler uma chamada, e métricas definidas antes de qualquer corrida com modelo.

O repositório tem as interfaces, as ferramentas falsas, o loop e o dataset versionado de avaliação. Os números do estudo entram no PR4, depois da medição.

## Objetivo

Comparar três estratégias de tool use sobre a mesma interface de LLM:

1. ReAct: o modelo escreve um traço com `Action` e `Action Input`.
2. Function calling nativo: o modelo devolve a chamada no formato estruturado do provedor.
3. Plano JSON: o modelo lista os passos; o loop executa a lista e só então pede a resposta final.

As tarefas cobrem três situações: uma ferramenta, duas ferramentas em sequência, e abstenção (nenhuma ferramenta deve ser chamada). A ferramenta `weather` é uma distração fictícia: fica exposta no catálogo e não entra na sequência esperada do dataset v1.

Métricas previstas para o PR3, sempre calculadas a partir de uma corrida registrada:

- acurácia de seleção da ferramenta
- correção dos argumentos
- chamadas desnecessárias
- correção da resposta final
- latência
- tokens e custo estimados

## Hipótese

A hipótese, ainda sem medição, é que o function calling nativo preserva melhor a forma dos argumentos do que o traço ReAct em texto livre, e que o plano JSON reduz chamadas desnecessárias nas tarefas de abstenção porque os passos ficam explícitos antes da execução. ReAct pode perder uma chamada por erro de formato. O plano JSON pode inserir um passo que a tarefa não pede. O PR4 mede essas diferenças.

## Metodologia planejada

- Cliente de LLM independente do provedor. A suíte usa um cliente falso com respostas roteirizadas. Sem a próxima resposta no roteiro, o cliente falha em vez de completar o texto.
- Três estratégias sobre esse cliente: `ReactToolStrategy`, `NativeToolStrategy` e `JsonPlanStrategy`. ReAct e o plano JSON pedem texto. A estratégia nativa envia o schema das ferramentas.
- Ferramentas falsas com esquema tipado: `lookup` (base fixa), `calculator` (aritmética por AST, sem `eval`) e `weather` (previsão fictícia). Consulta desconhecida devolve `not_found`.
- Fixture curta de 11 tarefas em `src/tool_calling_study/tasks/fixtures.json`, usada pelo loop e pelos testes do scaffold. O gabarito fechado do estudo é o dataset v1, descrito abaixo.
- O loop executa a estratégia, chama as ferramentas falsas e registra ferramenta, argumentos, resposta final e chamadas desnecessárias. Chamada desnecessária é ferramenta fora do gabarito ou listada como proibida. Repetir uma ferramenta esperada fica para a métrica do PR3.
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

O harness que percorre esse dataset e calcula as métricas fica no PR3.

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

## Roadmap

- PR3: harness de avaliação e métricas (seleção, argumentos, chamada desnecessária, resposta, latência, tokens e custo).
- PR4: experimentos com modelos reais e relatório.

## Layout

- `data/eval/tasks.v1.jsonl`: dataset versionado e o gabarito fechado.
- `src/tool_calling_study/llm`: interface, cliente falso e provedores com import tardio.
- `src/tool_calling_study/strategies`: parsers e as três estratégias.
- `src/tool_calling_study/tools`: esquemas e ferramentas falsas.
- `src/tool_calling_study/tasks`: fixture curta do loop e `load_eval_tasks`.
- `src/tool_calling_study/agent`: loop que grava o rastro.
- `src/tool_calling_study/checks.py`: detecção de chamada desnecessária.
