"""Stand-ins for third-party packages, installed ONLY when the real ones are missing.

They let the routing tests run on a machine without network access (or without the provider SDKs). When the real
packages are installed (pip install -r requirements.txt) these stubs do nothing.
"""
import importlib.util
import os
import sys
import tempfile
import types

# SAFETY: tests must never touch your real data. Force a throwaway database no matter what the environment says.
os.environ["LLMHUB_TELEMETRY_DB"] = os.path.join(tempfile.mkdtemp(prefix="llmhub-tests-"), "test.sqlite3")


def _missing(name):
    try:
        return importlib.util.find_spec(name) is None
    except (ImportError, ValueError):
        return True


def install():
    if _missing("openai"):
        m = types.ModuleType("openai"); m.OpenAI = object; sys.modules["openai"] = m
    if _missing("anthropic"):
        m = types.ModuleType("anthropic"); m.Anthropic = object; sys.modules["anthropic"] = m
    if _missing("google.genai"):
        google = sys.modules.get("google") or types.ModuleType("google")
        genai = types.ModuleType("google.genai")
        gtypes = types.ModuleType("google.genai.types")

        class Part:
            def __init__(self, text=None, data=None, mime_type=None):
                self.text, self.data, self.mime_type = text, data, mime_type

            @classmethod
            def from_text(cls, text):
                return cls(text=text)

            @classmethod
            def from_bytes(cls, data, mime_type):
                return cls(data=data, mime_type=mime_type)

        class Content:
            def __init__(self, role=None, parts=None):
                self.role, self.parts = role, parts

        class GenerateContentConfig:
            def __init__(self, **kw):
                self.__dict__.update(kw)

        gtypes.Part, gtypes.Content, gtypes.GenerateContentConfig = Part, Content, GenerateContentConfig
        genai.types, genai.Client = gtypes, object
        google.genai = genai
        sys.modules.update({"google": google, "google.genai": genai, "google.genai.types": gtypes})
    if _missing("keyring"):
        m = types.ModuleType("keyring")
        store = {}
        m.get_password = lambda s, u: store.get((s, u))
        m.set_password = lambda s, u, p: store.__setitem__((s, u), p)
        m.delete_password = lambda s, u: store.pop((s, u), None)
        sys.modules["keyring"] = m
    if _missing("fastapi"):
        fa = types.ModuleType("fastapi")

        class HTTPException(Exception):
            def __init__(self, status_code, detail=None):
                super().__init__(detail)
                self.status_code, self.detail = status_code, detail

        class FastAPI:
            def __init__(self, *a, **k): pass
            def add_middleware(self, *a, **k): pass

            def _route(self, *a, **k):
                return lambda f: f
            get = post = put = delete = _route

        class Response:
            def __init__(self, content=b"", media_type=None, headers=None):
                self.body, self.media_type, self.headers = content, media_type, headers or {}

        fa.FastAPI, fa.HTTPException, fa.Response = FastAPI, HTTPException, Response
        mw = types.ModuleType("fastapi.middleware"); cors = types.ModuleType("fastapi.middleware.cors")
        cors.CORSMiddleware = object
        sys.modules.update({"fastapi": fa, "fastapi.middleware": mw, "fastapi.middleware.cors": cors})
    if _missing("pydantic"):
        pd = types.ModuleType("pydantic")
        MISSING = object()

        class _F:
            def __init__(self, default=MISSING, default_factory=None, **k):
                self.default, self.default_factory = default, default_factory

        def Field(default=MISSING, default_factory=None, **k):
            return _F(default, default_factory)

        class BaseModel:
            def __init__(self, **kw):
                ann = {}
                for c in reversed(type(self).__mro__):
                    ann.update(getattr(c, "__annotations__", {}))
                for name in ann:
                    if name in kw:
                        v = kw[name]
                    else:
                        d = getattr(type(self), name, MISSING)
                        if isinstance(d, _F):
                            v = d.default_factory() if d.default_factory else d.default
                        elif d is MISSING:
                            raise TypeError(f"missing field {name}")
                        else:
                            v = d
                    setattr(self, name, v)

            def model_dump(self):
                def conv(x):
                    if isinstance(x, BaseModel): return x.model_dump()
                    if isinstance(x, list): return [conv(i) for i in x]
                    if isinstance(x, dict): return {k: conv(v) for k, v in x.items()}
                    return x
                return {k: conv(v) for k, v in self.__dict__.items()}

        pd.BaseModel, pd.Field = BaseModel, Field
        sys.modules["pydantic"] = pd
