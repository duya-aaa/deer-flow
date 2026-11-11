"""Credential-bearing API calls must reject redirects before a second request."""

import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from skill_loader import load  # noqa: E402

vid = load("video-generation")


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@pytest.mark.parametrize(
    "operation",
    ["minimax_poll", "gemini_poll", "retrieve", "minimax_generate", "gemini_generate"],
)
def test_api_redirect_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: int, operation: str
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-key")
    monkeypatch.setenv("MINIMAX_API_KEY", "synthetic-key")
    monkeypatch.delenv("MINIMAX_VIDEO_DURATION", raising=False)
    monkeypatch.delenv("MINIMAX_VIDEO_RESOLUTION", raising=False)
    seen = []

    def send(
        self: requests.adapters.HTTPAdapter,
        request: requests.PreparedRequest,
        **kwargs: object,
    ) -> requests.Response:
        seen.append(request)
        if len(seen) > 1:
            pytest.fail("API redirect must not issue a second request")
        response = requests.Response()
        response.status_code = status
        response.headers["Location"] = "https://redirect.example/target"
        response._content = b"{}"
        response.request = request
        response.url = request.url
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    output = str(tmp_path / "video.mp4")
    calls = {
        "minimax_poll": lambda: vid._poll_video_task(
            "https://api.minimaxi.com", "Bearer synthetic-key", "task"
        ),
        "gemini_poll": lambda: vid._poll_gemini_operation(
            "synthetic-key", "operations/test"
        ),
        "retrieve": lambda: vid._retrieve_file_url(
            "https://api.minimaxi.com", "Bearer synthetic-key", "file"
        ),
        "minimax_generate": lambda: vid._generate_video_minimax("prompt", [], output),
        "gemini_generate": lambda: vid._generate_video_gemini("prompt", [], output),
    }
    with pytest.raises(requests.HTTPError, match="redirect"):
        calls[operation]()
    assert len(seen) == 1
    assert not Path(output).exists()

