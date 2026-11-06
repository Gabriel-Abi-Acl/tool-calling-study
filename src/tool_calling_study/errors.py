# Erros do estudo. O loop converte falha de parse em registro, em vez de derrubar a bateria.
# Chave ausente e SDK ausente são erros distintos para o runner não tentar a rede.


class StudyError(Exception):
    """Base dos erros previstos pelo runner."""


class ParseError(StudyError):
    """Saída do modelo fora do formato da estratégia."""


class ConfigError(StudyError):
    """Variável de ambiente ausente ou com valor fora do conjunto aceito."""


class MissingCredential(StudyError):
    """Provedor ao vivo chamado sem a chave correspondente no ambiente."""


class ProviderNotInstalled(StudyError):
    """SDK do provedor não está instalado. O import só ocorre na chamada ao vivo."""


class ToolError(StudyError):
    """A ferramenta rejeitou a entrada. O loop registra a falha e segue."""
