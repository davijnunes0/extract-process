import logging
import os
import time
from dataclasses import dataclass
from threading import Lock
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlparse, urlunparse

import requests

logger = logging.getLogger(__name__)

class AIClientError(RuntimeError):
     """Erro base do cliente de IA."""

class AIAuthError(AIClientError):
    """Erro relacionado á autenticação"""


# O decorador @dataclasss(frozen=True) em Python transforma uma classe em uma estrutura de dados otimizada (com métodos __init__ e __repr__ gerados
# automaticamente) e "torna os objetos imutáveis"
@dataclass(frozen=True)
class AiClientConfig:
    base_url: str
    email: str | None
    password: str | None
    bearer_token: str | None
    timeout: int
    chat_completions_path: str
    login_path: str
    login_retries: int
    login_backoff_sec: float
    renew_delay_sec: float

class AiClient:
    """
        Cliente HTTP compátivel com:

        client.chat.completions.create(
            model="...",
            messages=[...]
        )

        Suporta:
        - token via parâmetro bearer_token
        - token na UR: ?token=...
        - token via variável OPENAI_BEARER_TOKEN
        - login via email/senha em /api/v1qauths/signin
        - renovação de token após 401
    """

    def __init__(
        self,
        base_url: str,
        email: str | None = None,
        password: str | None = None,
        bearer_token: str | None = None,
        timeout: int = 30,
        chat_completions_path: str = "/v1/chat/completions",
    ):
        clean_base_url, token_from_url = self._extract_token_from_url(base_url)

        self.config = AiClientConfig(
            base_url=clean_base_url.rstrip("/"),
            email=(
                email
                or os.getenv("OPENAI_BEARER_EMAIL")
                or os.getenv("OPENAI_API_USERNAME")
            ),
            password=(
                password
                or os.getenv("OPENAI_BEARER_PASSWORD")
                or os.getenv("OPENAI_API_PASSWORD")
            ),
            bearer_token=bearer_token or os.getenv("OPENAI_BEARER_TOKEN") or token_from_url,
            timeout=int(
                timeout
                or os.getenv("OPENAI_BEARER_TIMEOUT")
                or os.getenv("OPENAI_TIMEOUT_SECONDS")
                or 30
            ),
            chat_completions_path=(
                chat_completions_path
                or os.getenv("OPENAI_CHAT_COMPLETIONS_PATH")
                or os.getenv("OPENAI_CHAT_PATH")
                or "/api/chat"
            ),
            login_path="/api/v1/auths/signin",
            login_retries=int(os.getenv("OPENAI_LOGIN_RETRIES", 3)),
            login_backoff_sec=float(os.getenv("OPENAI_LOGIN_BACKOFF_SEC", 15)),
            renew_delay_sec=float(os.getenv("OPENAI_401_RENEW_DELAY_SEC", 5)),
        )

        self.session = requests.Session()
        self._cached_token: str | None = None
        self._token_lock = Lock()

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create_completion)
        )

    @staticmethod
    def _extract_token_from_url(base_url: str) -> tuple[str, str | None]:
        """
            Extrai token de uma URL no formato:
                https://exemplo.com?token=JWT
                https://exemplo.com?access_token=JWT

            Retorna:
                (url_sem_query,token)
        """

        if not base_url:
            return "", None

        # O método dividirá ssa string nos seguintes atributos:
        """
            scheme      -> https
            netloc      -> www.site.com.br:80
            path        -> /categoria/produto
            params      ->
            query       -> id=1234
            fragment    -> tudo que vem após a hashtag #
        """
        parsed = urlparse(base_url)

        if not parsed.query:
            return base_url, None

        # Cria o dicionário, os valores sempre serão lista, mesmo que haja apenas um valor para aquela chave.
        query = parse_qs(parsed.query)
        token_list = query.get("token", "") or query.get("access_token", "")

        if not token_list:
            return base_url, None

        token = token_list[0]

        # Este método faz o caminho reverso do urlparse. Ele pega esses 6 partes separadas (agora com a query vazia) e costura tudo de volta em uma única string
        # perfeitamente formatada

        clean_url = urlunparse(parsed._replace(query=""))

        return clean_url, token

    def _build_url(self, path: str) -> str:
        return f"{self.config.base_url}/{path.lstrip('/')}"

    def _get_token(self) -> str | None:
        """
            Retorna um token válido.

            Ordem de prioridade:
            1. Token em cache
            2. Token fornecido por parâmetro, URL ou variável de ambiente
            3. Login com email/senha
        """
        with self._token_lock:
            if self._cached_token:
                return self._cached_token

            if self.config.bearer_token:
                self._cached_token = self.config.bearer_token
                return self._cached_token

            if not self.config.email or not self.config.password:
                return None

            self._cached_token = self._login()
            return self._cached_token

    # Autenticação
    def _login(self) -> str | None:
        signin_url = self._build_url(self.config.login_path)
        payload = {
            "email": self.config.email,
            "password": self.config.password,
        }

        last_error: Exception | None = None
        for attempt in range(1, self.config.login_retries + 1):
            response = self.session.post(
                signin_url,
                json=payload,
                timeout=self.config.timeout,
            )

            if self._is_rate_limited(response):
                last_error = AIAuthError(
                    f"Rate limit ao fazer login: {self._response_error(response)}"
                )
                self._sleep_before_login_retry(attempt)
                continue

            if not response.ok:
                raise AIAuthError(
                    f"Erro ao fazer login ({response.status_code}): "
                    f"{self._response_error(response)}"
                )

            data = response.json()
            token = data.get("token") or data.get("access_token")

            if not token:
                raise AIAuthError("Token não encontrado na resposta do login.")

            return token

        raise last_error or AIAuthError("Falha ao fazer login")

    def _reset_token(self) -> None:
        with self._token_lock:
            self._cached_token = None

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}

        try:
            token = self._get_token()
        except Exception as exc:
            logger.warning("Erro ao obter token: %s", exc)
            token = None

        if token:
            headers["Authorization"] = f"Bearer {token}"

        return headers

    def create_completion(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float = 1,
            max_retries: int = 1,
            **kwargs: Any,
    ) -> SimpleNamespace:
        """
        Chama o endpoint de chat completions.

        Retorna um objeto compatível com:

            response.choices[0].message.content
        """
        if not self.config.base_url:
            raise AIClientError("base_url não configurado para AIClient.")

        url = self._build_url(self.config.chat_completions_path)

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
            **kwargs,
        }

        headers = self._build_headers()

        for attempt in range(max_retries + 1):
            response = self.session.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout,
            )

            if response.ok:
                return self._parse_completion_response(response)

            can_retry_auth = response.status_code == 401 and attempt < max_retries

            if can_retry_auth:
                self._wait_before_token_renew()
                self._reset_token()
                headers = self._build_headers()
                continue

            raise AIClientError(
                f"Erro na requisição ({response.status_code}): "
                f"{self._response_error(response)}"
            )

        raise AIClientError("Falha inesperada ao chamar chat completions.")


    @staticmethod
    def _parse_completion_response(response: requests.Response) -> SimpleNamespace:
        try:
            data = response.json()
        except ValueError as exc:
            raise AIClientError(
                f"Resposta 2xx, mas corpo não é JSON "
                f"(status={response.status_code}, body={response.text!r})"
            ) from exc

        if not isinstance(data, dict):
            raise AIClientError(
                f"Resposta JSON inesperada "
                f"(status={response.status_code}, body={response.text!r}, parsed={data!r})"
            )

        choices = []

        if "choices" in data:
            for choice in data.get("choices", []):
                message = choice.get("message", {})
                content = message.get("content", "")

                choices.append(
                    SimpleNamespace(
                        message=SimpleNamespace(content=content)
                    )
                )

            return SimpleNamespace(choices=choices, raw=data)

        message = data.get("message")
        if isinstance(message, dict):
            content = message.get("content", "")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=content)
                    )
                ],
                raw=data,
            )

        content = data.get("response")
        if isinstance(content, str):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=content)
                    )
                ],
                raw=data,
            )

        raise AIClientError(
            f"Formato de resposta não suportado "
            f"(status={response.status_code}, body={response.text!r}, parsed={data!r})"
        )

    @staticmethod
    def _response_error(response: requests.Response) -> str | None:
        try:
            return response.json()
        except ValueError:
            return response.text

    @staticmethod
    def _is_rate_limited(response: requests.Response) -> bool:
        body = response.text or ""

        return (
            response.status_code == 429
            or "rate limit" in body.lower()
        )

    def _sleep_before_login_retry(self, attempt: int) -> None:
        if attempt >= self.config.login_retries:
            return

        wait_seconds = self.config.login_backoff_sec * attempt

        logger.warning(
            "Rate limit no login. Aguardando %.0fs antes de tentar novamente "
            "(%d/%d).",
            wait_seconds,
            attempt,
            self.config.login_retries,
        )

        time.sleep(wait_seconds)

    def _wait_before_token_renew(self) -> None:
        if self.config.renew_delay_sec <= 0:
            return

        logger.info(
            "401 recebido. Aguardando %.0fs antes de renovar token.",
            self.config.renew_delay_sec,
        )

        time.sleep(float(self.config.renew_delay_sec))


    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "AIClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
