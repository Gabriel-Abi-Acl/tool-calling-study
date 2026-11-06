# tool-calling-study

Estudo comparativo de estratégias de uso de ferramentas em um loop de agente.
O recorte é metodológico: as mesmas tarefas sintéticas, três formas de pedir e ler uma chamada, e métricas definidas antes de qualquer corrida com modelo.

Este repositório está no PR1. Ele entrega interfaces, ferramentas falsas, um loop e testes offline. Os números do estudo entram no PR4, depois da medição.

## Objetivo

Comparar três estratégias de tool use sobre a mesma interface de LLM:

1. ReAct: o modelo escreve um traço com `Action` e `Action Input`.
2. Function calling nativo: o modelo devolve a chamada no formato estruturado do provedor.
3. Plano JSON: o modelo lista os passos; o loop executa a lista e só então pede a resposta final.

As tarefas cobrem três situações: uma ferramenta, duas ferramentas em sequência, e abstenção (nenhuma ferramenta deve ser chamada). A ferramenta `weather` é uma distração fictícia: em parte das tarefas ela está proibida.

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
- Fixture provisória de 11 tarefas em `src/tool_calling_study/tasks/fixtures.json`: 4 de uma ferramenta, 3 de duas, 4 de abstenção. Populações e previsões são constantes redondas deste estudo, não censo nem clima real.
- O loop executa a estratégia, chama as ferramentas falsas e registra ferramenta, argumentos, resposta final e chamadas desnecessárias. Chamada desnecessária é ferramenta fora do gabarito ou listada como proibida. Repetir uma ferramenta esperada fica para a métrica do PR3.
- `reference_answer` é o texto determinístico da última ferramenta da cadeia. O critério de resposta livre e o gabarito fechado ficam no PR2.

Os enunciados das tarefas estão em português. Os protocolos (`Action`, `Action Input`, `Final Answer`, o objeto de plano e os schemas) estão em inglês, no formato usado por essas estratégias.

Um turno ReAct válido tem uma `Action` e um `Action Input` em JSON, ou um `Final Answer`. O plano válido é um objeto `{"steps": [...], "final_answer": "..."}`. Argumentos que não são objeto JSON geram erro de parse, registrado pelo loop como `stopped_reason=parse_error`.

## Como rodar

Requisitos: Python 3.11+ e [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

A configuração é por variável de ambiente. `.env.example` lista os nomes; os valores de chave ficam vazios no git. Para um experimento local posterior, exporte as variáveis no ambiente. O arquivo `.env` está no `.gitignore`.

`LLM_PROVIDER=fake` é o padrão. `openai` e `anthropic` importam o SDK correspondente só dentro da primeira chamada ao vivo. Sem a chave, a chamada para antes do import. Sem o pacote do SDK instalado no ambiente, a chamada levanta `ProviderNotInstalled` e não abre rede. Esses SDKs não fazem parte das dependências de desenvolvimento deste PR.

O CI usa GitHub Actions, instala uv pelo release `astral-sh/setup-uv@v10.3.3` e roda ruff e pytest.

## Roadmap

- PR2: dataset completo e gabarito (argumentos esperados revisados e critério da resposta final).
- PR3: avaliação e métricas (seleção, argumentos, chamada desnecessária, resposta, latência, tokens e custo).
- PR4: experimentos com modelos reais e relatório.

## Layout

- `src/tool_calling_study/llm`: interface, cliente falso e provedores com import tardio.
- `src/tool_calling_study/strategies`: parsers e as três estratégias.
- `src/tool_calling_study/tools`: esquemas e ferramentas falsas.
- `src/tool_calling_study/tasks`: fixture sintética.
- `src/tool_calling_study/agent`: loop que grava o rastro.
- `src/tool_calling_study/checks.py`: detecção de chamada desnecessária.
