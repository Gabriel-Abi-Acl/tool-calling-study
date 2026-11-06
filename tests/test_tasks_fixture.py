# A fixture do PR1 fica pequena e sintética: 8 a 12 tarefas, com os três kinds.
# O clima aparece numa tarefa e fica proibido nas outras, como distração.

from collections import Counter

from tool_calling_study.tasks import load_tasks


def test_placeholder_fixture_covers_single_multi_and_abstain():
    tasks = load_tasks()
    kinds = Counter(task.kind for task in tasks)
    assert 8 <= len(tasks) <= 12
    assert kinds["single"] >= 1
    assert kinds["multi"] >= 1
    assert kinds["abstain"] >= 1
    assert sum(kinds.values()) == len(tasks)
    assert all(task.prompt for task in tasks)
    weather_tasks = [task for task in tasks if "weather" in task.expected_tools]
    distracted = [task for task in tasks if "weather" in task.forbidden_tools]
    assert len(weather_tasks) == 1
    assert len(distracted) == len(tasks) - 1
