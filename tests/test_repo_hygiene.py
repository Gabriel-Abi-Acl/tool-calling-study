# O repositório versiona o exemplo de ambiente com chave vazia e o pin exato do setup-uv.
# Um tag móvel de major do setup-uv já respondeu 404; o workflow fica no release v10.3.3.

from pathlib import Path


def test_env_example_keeps_secrets_empty_and_gitignore_covers_dotenv():
    example = Path(".env.example").read_text(encoding="utf-8")
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        assert f"{name}=" in example
        value = next(
            line.split("=", 1)[1] for line in example.splitlines() if line.startswith(f"{name}=")
        )
        assert value.strip() == ""
    assert ".env" in Path(".gitignore").read_text(encoding="utf-8").splitlines()


def test_ci_pins_setup_uv_to_an_existing_patch_tag():
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "astral-sh/setup-uv@v10.3.3" in workflow
    assert "setup-uv@v10\n" not in workflow
    assert "setup-uv@v1\n" not in workflow
