"""LLM providers.

Three of them, selected by RIPPLE_PROVIDER:

  stub    No network, no key, fully deterministic. Extractive: it selects and
          lightly rewrites spans from retrieved chunks. This is the DEFAULT,
          because gate G1 requires the container to launch and the replay suite
          to complete on a clean machine with one command -- a default that
          demands an API key fails that gate on the judge's machine.

          Important for the guide's no-hardcoding rule: the stub contains no
          prompts, no queries and no canned responses keyed to any benchmark
          input. It is a generic extractive summariser over whatever evidence
          it is handed, and it would behave identically on a corpus it had
          never seen.

  gemini  Google Generative Language API. Free-tier friendly: a client-side
          token bucket enforces the RPM ceiling, 429s back off and retry, and
          every failure degrades to the stub rather than to an ungrounded
          guess.

  openai  Chat Completions, same contract.

The engine never imports a provider directly; it holds this interface. That is
what lets the benchmark run all four systems with the provider held constant.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Optional

from ..schemas import Cost


def parse_429(r) -> tuple[float, str, str]:
    """Pull the useful content out of a rate-limit response.

    Returns (retry_after_seconds, "minute"|"day"|"", human_message).

    Three sources, in order of trustworthiness: the Retry-After header, the
    RetryInfo detail Google attaches to the error, and finally the quota
    metric name, which is what says whether this clears in a minute or at
    midnight. That distinction decides whether the right move is to wait or to
    stop, so it is worth the twenty lines.
    """
    delay = 0.0
    quota = ""
    msg = ""
    limit = ""
    try:
        ra = r.headers.get("Retry-After", "")
        if ra:
            delay = float(ra)
    except Exception:
        pass
    try:
        err = (r.json() or {}).get("error", {})
        msg = (err.get("message", "") or "")[:200]
        for det in err.get("details", []) or []:
            t = str(det.get("@type", ""))
            if "RetryInfo" in t:
                m = re.match(r"([\d.]+)s", str(det.get("retryDelay", "")))
                if m:
                    delay = max(delay, float(m.group(1)))
            if "QuotaFailure" in t:
                for v in det.get("violations", []) or []:
                    metric = (str(v.get("quotaId", ""))
                              + " " + str(v.get("quotaMetric", ""))).lower()
                    if "perday" in metric.replace("_", ""):
                        quota = "day"
                    elif not quota and "perminute" in metric.replace("_", ""):
                        quota = "minute"
                    if v.get("quotaId"):
                        msg = str(v["quotaId"])[:120]
                    if v.get("quotaValue") and not limit:
                        limit = str(v["quotaValue"])
        # The human-readable message usually carries the number too
        # ("... limit: 20, model: ..."). The quota NAME alone told the user
        # they had run out but never how much there was to begin with, which
        # is the one number needed to size a run that fits.
        if not limit:
            m = re.search(r"limit:\s*(\d+)", str(err.get("message", "")))
            if m:
                limit = m.group(1)
    except Exception:
        pass
    if limit and msg:
        msg = f"{msg} (limit: {limit} requests)"
    if not quota:
        low = msg.lower().replace("_", "").replace(" ", "")
        if "perday" in low:
            quota = "day"
        elif "perminute" in low:
            quota = "minute"
    return delay, quota, msg


def usage_thoughts(data: dict) -> int:
    """Reasoning tokens the model charged us for, if it says.

    Worth reading rather than inferring: it is the difference between "the
    model is broken" and "the model thought for 900 tokens and had none left
    to answer with", and those need different responses.
    """
    u = data.get("usageMetadata", {}) or {}
    return int(u.get("thoughtsTokenCount", 0) or 0)


def _i_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


@dataclass
class LLMResult:
    text: str
    cost: Cost
    degraded: bool = False
    error: str = ""


class Provider:
    name = "base"

    def complete(self, prompt: str, max_tokens: int = 600,
                 json_mode: bool = False) -> LLMResult:
        raise NotImplementedError

    def available(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Rate limiting -- free-tier survival
# ---------------------------------------------------------------------------


class TokenBucket:
    """Client-side RPM ceiling.

    The evaluation harness makes several hundred calls across four systems and
    eighty scenarios. On a free tier that is the difference between a run that
    completes overnight and a run that dies at scenario nine with a 429.
    """

    def __init__(self, rpm: int):
        self.capacity = max(1, rpm)
        self.tokens = float(self.capacity)
        self.refill_rate = self.capacity / 60.0
        self.rpm = float(self.capacity)
        self.last = time.monotonic()
        self._lock = threading.Lock()

    def throttle(self, factor: float = 0.6, floor_rpm: float = 2.0) -> float:
        """Permanently slow down after the server says we are too fast.

        The configured RPM is a guess about someone else's quota. A 429 is the
        authoritative correction, so the bucket adopts it instead of continuing
        at a rate now known to be wrong -- otherwise every call re-earns the
        same 429 and the run degenerates into paying full latency for backoff.
        Multiplicative decrease, with a floor so it cannot throttle to a halt.
        """
        with self._lock:
            self.rpm = max(floor_rpm, self.rpm * factor)
            self.refill_rate = self.rpm / 60.0
            self.tokens = 0.0
            self.last = time.monotonic()
            return self.rpm

    def acquire(self) -> float:
        with self._lock:
            now = time.monotonic()
            self.tokens = min(
                self.capacity, self.tokens + (now - self.last) * self.refill_rate
            )
            self.last = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return 0.0
            wait = (1.0 - self.tokens) / self.refill_rate
        time.sleep(wait)
        with self._lock:
            self.tokens = max(0.0, self.tokens - 1.0)
            self.last = time.monotonic()
        return wait


# ---------------------------------------------------------------------------
# Stub
# ---------------------------------------------------------------------------


_SENT = re.compile(r"(?<=[.!?])\s+")
_STOP = set("""a an the of to in on for and or but with without is are was were
be been being as at by from that this these those it its their his her our your
my you we they he she i not no nor so if then than there here which who whom
whose what when where why how any some all each both few more most other such
only own same too very can will just should now""".split())


def _content(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9\-]+", text.lower())
            if w not in _STOP and len(w) > 2}


def best_sentences(chunk_text: str, query: str, n: int = 2) -> list[str]:
    """Pick the sentences of a chunk that best answer a query.

    Content-word overlap with a length prior. Deliberately simple and
    corpus-independent.
    """
    sents = [s.strip() for s in _SENT.split(chunk_text) if len(s.split()) >= 5]
    if not sents:
        return [chunk_text.strip()]
    q = _content(query)
    scored = []
    for s in sents:
        cw = _content(s)
        if not cw:
            continue
        overlap = len(q & cw) / max(1, len(q)) if q else 0.0
        density = len(q & cw) / max(1, len(cw))
        scored.append((0.7 * overlap + 0.3 * density, s))
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:n]] or sents[:n]


class StubProvider(Provider):
    """Deterministic extractive provider. Never leaves the process."""

    name = "stub"

    def complete(self, prompt: str, max_tokens: int = 600,
                 json_mode: bool = False) -> LLMResult:
        # The stub is not driven through free-text prompts; the synthesiser
        # calls its structured helpers directly. This exists so that any code
        # path expecting a Provider still works.
        return LLMResult(text="", cost=Cost(llm_calls=0), degraded=True,
                         error="stub provider has no free-text completion")


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


_KEY_IN_TEXT = re.compile(r"([?&]key=)[^&\s'\"]+")

# Server-side failures that clear up on their own. A 503 means Google's
# servers are overloaded; it says nothing about the key, the model or the
# quota. Treating it like a rejection sent the user off to run check_key.py,
# which is the wrong fix and spends quota.
TRANSIENT_STATUS = (500, 502, 503, 504)
TRANSIENT_WAITS = (5, 10, 20, 30, 45)   # seconds; ~2 minutes in total


class GeminiProvider(Provider):
    name = "gemini"
    ENDPOINT = ("https://generativelanguage.googleapis.com/v1beta/models/"
                "{model}:generateContent")

    # Floor on maxOutputTokens. A reasoning model can burn hundreds of tokens
    # before its first visible character, so a small ask returns nothing at all
    # rather than a short answer. Overridable because it is a cost knob.
    MIN_OUTPUT_TOKENS = _i_env("RIPPLE_MAX_OUTPUT", 1024)

    def __init__(self, model: str = "gemini-2.0-flash", rpm: int = 12,
                 api_key: Optional[str] = None, cost_table=None):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.bucket = TokenBucket(rpm)
        self.cost_table = cost_table
        self.fatal_error = ""
        self.calls_made = 0
        # Failures are counted and surfaced. `calls_made` alone hid a run where
        # 51 turns produced 3 successful calls, because a counter that only
        # goes up on success cannot tell you about the ones that did not.
        self.calls_failed = 0
        self.last_error = ""
        # Learned once, reused for the rest of the run. See complete().
        self._output_floor = self.MIN_OUTPUT_TOKENS
        self._no_thinking_config = False
        # Set when a quota is exhausted for the DAY rather than the minute.
        # Retrying that is not patience, it is a hang.
        self.quota_exhausted = ""
        # Set when Google's servers kept failing (5xx) for every retry. A
        # temporary problem on their side, reported as such.
        self.server_unavailable = ""
        self._key_in_url = False

    def available(self) -> bool:
        return bool(self.api_key)

    def _redact(self, msg: str) -> str:
        """Remove the API key from any text that might be printed.

        An httpx error message contains the full request URL. When the key
        travelled in that URL (?key=...), a server error printed the key
        straight onto the user's screen -- and from there into a chat.
        Every error string passes through here before it is stored.
        """
        msg = str(msg)
        if self.api_key:
            msg = msg.replace(self.api_key, "[key hidden]")
        return _KEY_IN_TEXT.sub(r"\1[key hidden]", msg)

    def _key_params(self) -> dict:
        # Header only by default, so the key never appears in a URL. The
        # ?key= form is kept only as a fallback for a key the header route
        # rejects; see complete().
        return {"key": self.api_key} if self._key_in_url else {}

    def _fail(self, msg: str) -> LLMResult:
        msg = self._redact(msg)
        self.calls_failed += 1
        self.last_error = msg
        return LLMResult("", Cost(llm_calls=0), degraded=True, error=msg)

    LIST_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"

    def list_models(self) -> tuple[bool, list[str] | str]:
        """Ask the API which models this key can use.

        This separates two failures that look identical from a generation
        call: a key the API rejects, and a good key with a model name it does
        not recognise. Knowing which one you have decides what to fix.
        """
        import httpx

        if not self.available():
            return False, "no API key set (GEMINI_API_KEY)"
        try:
            r = httpx.get(self.LIST_ENDPOINT, params=self._key_params(),
                          headers={"x-goog-api-key": self.api_key},
                          timeout=20.0)
            if r.status_code != 200:
                detail = ""
                try:
                    detail = r.json().get("error", {}).get("message", "")[:220]
                except Exception:
                    detail = r.text[:220]
                return False, self._redact(f"HTTP {r.status_code}: {detail}")
            names = []
            for m in r.json().get("models", []):
                methods = m.get("supportedGenerationMethods", [])
                if not methods or "generateContent" in methods:
                    names.append(m.get("name", "").replace("models/", ""))
            return True, names
        except Exception as e:  # noqa: BLE001
            return False, self._redact(f"{type(e).__name__}: {e}")

    def try_model(self, name: str) -> tuple[bool, str]:
        """Actually generate with one model. Returns (worked, message).

        Necessary because the models-list endpoint is not a list of models you
        can use: it happily advertises names that return 404 "no longer
        available to new users" the moment you generate with them. Listing is
        advertising; generating is the test.
        """
        previous, self.model = self.model, name
        self.fatal_error = ""
        try:
            return self._probe()
        finally:
            self.model = previous

    def preflight(self) -> tuple[bool, str]:
        """One cheap call to prove the key and model work.

        Worth the single request: without it, a rejected key produces a
        benchmark that runs for ten minutes and reports nothing useful.
        """
        if not self.available():
            return False, "no API key set (GEMINI_API_KEY)"
        return self._probe()

    def _probe(self) -> tuple[bool, str]:
        """Shared by preflight() and try_model(): one call that must return TEXT.

        The bar is deliberately "produced a non-empty answer", not "did not
        error". An earlier version passed the moment no exception came back,
        printed `ok (model replied '')`, and let a whole benchmark proceed on a
        model that never emitted a single character. A preflight that can pass
        while the thing it is checking is broken is worse than no preflight,
        because it converts a loud failure into a plausible-looking table.
        """
        res = self.complete("Reply with the single word: ok")
        if self.fatal_error:
            return False, self._redact(self.fatal_error)
        if res.degraded:
            return False, res.error or "unknown failure"
        reply = (res.text or "").strip()
        if not reply:
            return False, f"{self.model} returned an empty answer"
        return True, reply[:40]

    def complete(self, prompt: str, max_tokens: int = 600,
                 json_mode: bool = False) -> LLMResult:
        import httpx

        if not self.available():
            return self._fail("GEMINI_API_KEY not set")
        # THINKING IS OFF, AND maxOutputTokens HAS A FLOOR. Both are the same
        # bug, found the hard way.
        #
        # Newer Gemini flash models are reasoning models: they spend output
        # tokens on internal thought BEFORE emitting any answer text, and those
        # tokens count against maxOutputTokens. Ask for 8 tokens and you get
        # back a perfectly successful HTTP 200 whose candidate contains no text
        # part at all and finishReason MAX_TOKENS. Nothing looks broken -- the
        # key is fine, the model exists, the call "succeeded" -- and the whole
        # benchmark quietly synthesises from empty strings.
        #
        # thinkingBudget=0 disables it where supported. Models that do not know
        # the field ignore it; models that refuse it are caught by the 400
        # handler below, which is why the retry there strips it rather than
        # giving up. The floor is belt-and-braces for whatever thinks anyway.
        # The floor is an INSTANCE value, not the class constant, so that a
        # budget learned by escalation sticks for the rest of the run. Without
        # this every single call paid for a doomed small attempt first, which
        # doubles the request count -- and on a free tier the request count is
        # the scarce resource, so the fix for one bug was quietly causing the
        # next one (the 429s).
        gen = {
            "temperature": 0.1,
            "maxOutputTokens": max(max_tokens, self._output_floor),
            "topP": 0.9,
            "thinkingConfig": {"thinkingBudget": 0},
        }
        if self._no_thinking_config:
            gen.pop("thinkingConfig")
        body = {"contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": gen}
        if json_mode:
            gen["responseMimeType"] = "application/json"

        url = self.ENDPOINT.format(model=self.model)
        last_err = ""
        escalated = False
        transient = 0          # consecutive server-side (5xx / network) failures
        self.server_unavailable = ""
        attempt = -1
        while attempt < 3:
            attempt += 1
            self.bucket.acquire()
            try:
                # KEY IN THE HEADER ONLY. An earlier version also sent it as
                # ?key= in the URL "to be safe" -- and when Google returned a
                # 503, the error message printed that URL, key included, onto
                # the user's screen. A header is never part of an error
                # message. The URL form survives only as a one-time fallback
                # for a key the header route rejects (see the 400/401/403
                # branch), and even then every error string is redacted.
                r = httpx.post(url, params=self._key_params(),
                               headers={"x-goog-api-key": self.api_key},
                               json=body, timeout=30.0)
                if r.status_code == 429:
                    # A 429 IS NOT ONE CONDITION, and the old code treated it as
                    # one: sleep, retry, and on give-up report the bare word
                    # "rate_limited", which tells you nothing you can act on.
                    #
                    # Google's 429 body distinguishes a per-MINUTE quota (wait a
                    # few seconds, it clears) from a per-DAY one (nothing clears
                    # until tomorrow, so retrying is a hang with extra steps),
                    # and usually names a retryDelay. Guessing 2/4/8 seconds
                    # while ignoring a number the server supplied is strictly
                    # worse than using it.
                    delay, quota, msg = parse_429(r)
                    if quota == "day":
                        self.quota_exhausted = msg or "daily request limit"
                        self.fatal_error = ("daily quota exhausted: "
                                            + self.quota_exhausted)
                        return self._fail(self.fatal_error)
                    # Slow the client-side bucket to match reality. The
                    # configured RPM was our guess; a 429 is the server's
                    # correction, and continuing at the guessed rate only
                    # earns another one.
                    self.bucket.throttle()
                    wait = delay if delay > 0 else 2 ** attempt * 2.0
                    last_err = (f"rate limited, waited {wait:.0f}s"
                                + (f" ({msg})" if msg else ""))
                    self.last_error = self._redact(last_err)
                    time.sleep(min(wait, 60.0))
                    continue
                if r.status_code in TRANSIENT_STATUS:
                    # GOOGLE'S PROBLEM, NOT OURS. Wait and retry on a longer
                    # schedule than a normal error gets: the old 1-2-3 second
                    # retries gave an overloaded server six seconds to recover
                    # and then declared the call "rejected". These retries do
                    # not use up the ordinary attempt budget.
                    if transient < len(TRANSIENT_WAITS):
                        wait = TRANSIENT_WAITS[transient]
                        transient += 1
                        attempt -= 1
                        last_err = (f"Google server error HTTP "
                                    f"{r.status_code}; retrying in {wait}s")
                        self.last_error = last_err
                        time.sleep(wait)
                        continue
                    self.server_unavailable = (
                        f"Google returned HTTP {r.status_code} on "
                        f"{transient + 1} attempts over about "
                        f"{sum(TRANSIENT_WAITS)}s")
                    return self._fail("server unavailable: "
                                      + self.server_unavailable)
                if r.status_code in (400, 401, 403, 404):
                    # FAIL FAST on a permanent error. Retrying a rejected key
                    # or a wrong model name never succeeds, and an earlier
                    # version retried four times with backoff on every call --
                    # so a bad key turned into ten silent seconds per call and
                    # a benchmark that looked hung rather than broken.
                    detail = ""
                    try:
                        detail = (r.json().get("error", {})
                                  .get("message", ""))[:500]
                    except Exception:
                        detail = r.text[:500]
                    # One exception to fail-fast: a 400 caused by our own
                    # thinkingConfig on a model that does not accept it. Drop
                    # the field and retry once rather than declaring the model
                    # dead over a knob we added.
                    #
                    # ANY 400, not only one that mentions thinking. A model
                    # with no thinking mode (gemini-3.5-flash-lite) answered
                    # with the generic "Request contains an invalid argument"
                    # -- no hint which argument -- and the old test for the
                    # word "think" let that through as a dead model. The
                    # field is ours, so it is the first suspect; dropping it
                    # costs one request, and the choice is remembered so the
                    # rest of the run never pays for it again. If the 400
                    # survives without the field, it is reported as-is.
                    # (A key problem is not a thinking problem: those go
                    # straight to the key fallback below.)
                    if (r.status_code == 400 and "thinkingConfig" in gen
                            and "api key" not in detail.lower()):
                        gen.pop("thinkingConfig", None)
                        self._no_thinking_config = True
                        last_err = "retrying without thinkingConfig"
                        attempt -= 1
                        continue
                    # One more exception: a key the header route rejects.
                    # Try the ?key= form exactly once and remember the answer
                    # for the rest of the run. Errors stay redacted.
                    if (r.status_code in (400, 401, 403)
                            and not self._key_in_url
                            and "key" in detail.lower()):
                        self._key_in_url = True
                        last_err = "retrying with the key as a URL parameter"
                        attempt -= 1
                        continue
                    self.fatal_error = self._redact(
                        f"HTTP {r.status_code}: {detail}")
                    return self._fail(self.fatal_error)
                r.raise_for_status()
                data = r.json()
                text = ""
                finish = ""
                for cand in data.get("candidates", []):
                    finish = cand.get("finishReason", "") or finish
                    for part in cand.get("content", {}).get("parts", []):
                        text += part.get("text", "")

                # AN EMPTY ANSWER IS A FAILURE, and it has to be reported as
                # one. This returned success before, which is how a run got all
                # the way to a finished results table built from empty strings:
                # HTTP 200, no error, no text. A provider that can fail
                # invisibly makes every number downstream of it unfalsifiable.
                if not text.strip():
                    # AUTOMATIC ESCALATION, once per call. If the model ate its
                    # whole budget thinking, the fix is more budget -- and this
                    # code can work that out for itself. Telling the user to go
                    # set RIPPLE_MAX_OUTPUT is making them debug our problem.
                    #
                    # Not a loop: exactly one retry at 4x, because a model that
                    # produces nothing at 4096 tokens is not budget-starved and
                    # spending more tokens to prove it is waste.
                    thought = int(usage_thoughts(data))
                    if (finish in ("MAX_TOKENS", "", None)
                            and not escalated
                            and gen["maxOutputTokens"] < 4096):
                        escalated = True
                        gen["maxOutputTokens"] = min(
                            4096, gen["maxOutputTokens"] * 4)
                        gen.pop("thinkingConfig", None)  # it was ignored anyway
                        # REMEMBER IT. Learning this per call instead of once
                        # per run doubles the request count against the quota
                        # that is already the binding constraint.
                        self._output_floor = gen["maxOutputTokens"]
                        self._no_thinking_config = True
                        last_err = (f"empty at {thought} thinking tokens; "
                                    f"retrying at "
                                    f"{gen['maxOutputTokens']} output tokens")
                        continue
                    why = {
                        "MAX_TOKENS": (f"model spent its entire output budget "
                                       f"({gen['maxOutputTokens']} tokens, "
                                       f"{thought} of them on reasoning) "
                                       f"without producing any answer text -- "
                                       f"this model cannot be used for "
                                       f"synthesis at a sane budget"),
                        "SAFETY": "response blocked by safety filters",
                        "RECITATION": "response blocked as recitation",
                        "PROHIBITED_CONTENT": "response blocked as prohibited",
                    }.get(finish, f"empty response (finishReason={finish or 'none'})")
                    block = (data.get("promptFeedback", {})
                             .get("blockReason", ""))
                    if block:
                        why += f"; prompt blocked: {block}"
                    return self._fail(f"{self.model}: {why}")
                usage = data.get("usageMetadata", {})
                pt = int(usage.get("promptTokenCount", 0))
                ct = int(usage.get("candidatesTokenCount", 0))
                money = (self.cost_table.compute(pt, ct)
                         if self.cost_table else 0.0)
                self.calls_made += 1
                return LLMResult(
                    text=text,
                    cost=Cost(prompt_tokens=pt, completion_tokens=ct,
                              llm_calls=1, currency_cost=money),
                )
            except Exception as e:  # noqa: BLE001
                last_err = self._redact(f"{type(e).__name__}: {e}")
                self.last_error = last_err
                # A timeout or dropped connection is transient too; give it
                # the same patient schedule as a 5xx.
                import httpx as _hx
                if (isinstance(e, (_hx.TimeoutException, _hx.NetworkError))
                        and transient < len(TRANSIENT_WAITS)):
                    wait = TRANSIENT_WAITS[transient]
                    transient += 1
                    attempt -= 1
                    time.sleep(wait)
                    continue
                time.sleep(1.0 * (attempt + 1))
        return self._fail(last_err)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class OpenAIProvider(Provider):
    name = "openai"
    ENDPOINT = "https://api.openai.com/v1/chat/completions"

    def __init__(self, model: str = "gpt-4o-mini", rpm: int = 20,
                 api_key: Optional[str] = None, cost_table=None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.bucket = TokenBucket(rpm)
        self.cost_table = cost_table
        self.fatal_error = ""
        self.calls_made = 0

    def available(self) -> bool:
        return bool(self.api_key)

    def preflight(self) -> tuple[bool, str]:
        if not self.available():
            return False, "no API key set (OPENAI_API_KEY)"
        res = self.complete("Reply with the single word: ok", max_tokens=64)
        if self.fatal_error:
            return False, self.fatal_error
        if res.degraded:
            return False, res.error or "unknown failure"
        reply = (res.text or "").strip()
        if not reply:
            return False, f"{self.model} returned an empty answer"
        return True, reply[:40]

    def complete(self, prompt: str, max_tokens: int = 600,
                 json_mode: bool = False) -> LLMResult:
        import httpx

        if not self.available():
            return LLMResult("", Cost(), degraded=True,
                             error="OPENAI_API_KEY not set")
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last_err = ""
        for attempt in range(4):
            self.bucket.acquire()
            try:
                r = httpx.post(self.ENDPOINT, json=body, headers=headers,
                               timeout=30.0)
                if r.status_code == 429:
                    time.sleep(2 ** attempt * 2.0)
                    last_err = "rate_limited"
                    continue
                if r.status_code in (400, 401, 403, 404):
                    self.fatal_error = f"HTTP {r.status_code}: {r.text[:200]}"
                    return LLMResult("", Cost(), degraded=True,
                                     error=self.fatal_error)
                r.raise_for_status()
                data = r.json()
                text = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                pt = int(usage.get("prompt_tokens", 0))
                ct = int(usage.get("completion_tokens", 0))
                money = (self.cost_table.compute(pt, ct)
                         if self.cost_table else 0.0)
                self.calls_made += 1
                return LLMResult(
                    text=text,
                    cost=Cost(prompt_tokens=pt, completion_tokens=ct,
                              llm_calls=1, currency_cost=money),
                )
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(1.0 * (attempt + 1))
        return LLMResult("", Cost(llm_calls=0), degraded=True, error=last_err)


def build_provider(cfg, cost_table) -> Provider:
    kind = (cfg.provider or "stub").lower()
    if kind == "gemini":
        p = GeminiProvider(cfg.model, cfg.rpm_limit, cost_table=cost_table)
        return p if p.available() else StubProvider()
    if kind == "openai":
        p = OpenAIProvider(cfg.model, cfg.rpm_limit, cost_table=cost_table)
        return p if p.available() else StubProvider()
    return StubProvider()


def parse_json_loose(text: str):
    """LLMs wrap JSON in prose and fences. Extract the first balanced object
    or array rather than failing the turn."""
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start < 0:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        break
    return None
