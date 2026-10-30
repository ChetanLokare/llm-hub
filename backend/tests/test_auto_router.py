import base64
import random
import unittest
from collections import Counter
from datetime import datetime, timezone

from _stubs import install
install()

from llm_hub.attachments import Part, Prepared  # noqa: E402
from llm_hub.auto_router import (AllAttemptsFailed, BudgetBlocked, NoCandidates, plan_attempts, run_chat)  # noqa: E402
from llm_hub.capabilities import Health, UnsupportedModality  # noqa: E402
from llm_hub.models import ModelInfo  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8


class Err(Exception):
    def __init__(self, msg, status=None):
        super().__init__(msg)
        self.status_code = status


class FP:
    def __init__(self, pid, models):
        self.provider_id, self.display_name, self._m = pid, pid.title(), models

    def list_models(self):
        return self._m


def M(pid, mid, ctx=100_000, billing="unknown", support=None, caps=("text",)):
    return ModelInfo(pid.title(), mid, mid, ctx, list(caps), "t", billing, support)


def image_prep():
    return Prepared([Part("a.png", "image", "image/png", PNG)])


def history(text="hello"):
    return [{"role": "user", "content": text}]


class Harness:
    """Runs run_chat with fake send(); `behave[(pid, mid)]` returns text or raises."""

    def __init__(self, views, behave=None):
        self.views, self.behave, self.calls, self.checks = views, behave or {}, [], {}
        self.health = Health()

    def record(self, pid, mid, cap, result, evidence):
        self.checks[(pid, mid, cap)] = {"result": result, "evidence": evidence,
                                        "checked_at": datetime.now(timezone.utc).isoformat()}

    def send(self, provider, model, messages):
        self.calls.append((provider.provider_id, model.model_id, messages[-1]["content"]))
        out = self.behave.get((provider.provider_id, model.model_id), "ok")
        if isinstance(out, BaseException):
            raise out
        if callable(out):
            out = out(messages)
        return out, None, [], "note"

    def run(self, prepared=None, **kw):
        args = dict(views=self.views, history=history(), prepared=prepared or Prepared(), selected=None, auto=True,
                    safe_fallback=True, prefer_free=False, has_key=lambda pid: True, send=self.send,
                    checks=self.checks, record_check=self.record, health=self.health, rng=random.Random(1))
        args.update(kw)
        return run_chat(**args)


class Ordering(unittest.TestCase):
    def test_no_vendor_or_model_name_bias(self):
        """With nothing known about the models, any of them can come first - the name decides nothing."""
        views = [FP("alpha", [M("alpha", "gpt-5-pro")]), FP("beta", [M("beta", "tiny-1b")]), FP("gamma", [M("gamma", "claude-opus")])]
        firsts = Counter()
        for seed in range(300):
            plan = plan_attempts(views, set(), set(), 0, {}, has_key=lambda p: True, health=Health(), recent={}, rng=random.Random(seed))
            firsts[plan[0].provider.provider_id] += 1
        self.assertEqual(set(firsts), {"alpha", "beta", "gamma"})
        self.assertTrue(all(c > 50 for c in firsts.values()), firsts)

    def test_least_used_provider_goes_first(self):
        views = [FP("a", [M("a", "m")]), FP("b", [M("b", "m")])]
        plan = plan_attempts(views, set(), set(), 0, {}, has_key=lambda p: True, health=Health(), recent={},
                             load={"a": 9_000_000}, rng=random.Random(0))
        self.assertEqual(plan[0].provider.provider_id, "b")

    def test_providers_are_interleaved_so_one_big_catalog_cannot_crowd_out_the_rest(self):
        views = [FP("big", [M("big", f"m{i}") for i in range(30)]), FP("small", [M("small", "only")])]
        for seed in range(20):
            plan = plan_attempts(views, set(), set(), 0, {}, has_key=lambda p: True, health=Health(), recent={}, rng=random.Random(seed))
            self.assertIn("small", {a.provider.provider_id for a in plan[:2]})

    def test_prefer_free_is_a_soft_order(self):
        views = [FP("a", [M("a", "paid1", billing="paid"), M("a", "free1", billing="free")]),
                 FP("b", [M("b", "unk", billing="unknown")])]
        plan = plan_attempts(views, set(), set(), 0, {}, has_key=lambda p: True, prefer_free=True, health=Health(), recent={}, rng=random.Random(2))
        self.assertEqual([a.billing for a in plan], ["free", "unknown", "paid"])  # nothing excluded
        plan = plan_attempts([FP("a", [M("a", "paid1", billing="paid")])], set(), set(), 0, {}, has_key=lambda p: True,
                             prefer_free=True, health=Health(), recent={})
        self.assertEqual(len(plan), 1)  # a paid model is still used when it is all there is

    def test_selected_model_is_first_when_not_auto(self):
        views = [FP("a", [M("a", "x", billing="free")]), FP("b", [M("b", "mine", billing="paid")])]
        plan = plan_attempts(views, set(), set(), 0, {}, has_key=lambda p: True, prefer_free=True, preferred=("b", "mine"),
                             health=Health(), recent={})
        self.assertEqual(plan[0].model.model_id, "mine")

    def test_typed_model_not_in_provider_list_is_still_honoured(self):
        views = [FP("a", [M("a", "listed")])]
        plan = plan_attempts(views, set(), set(), 0, {}, has_key=lambda p: True, preferred=("a", "brand-new"), health=Health(), recent={})
        self.assertEqual(plan[0].model.model_id, "brand-new")

    def test_recent_failure_moves_a_model_last_not_out(self):
        h = Health()
        h.penalize("a", "m", "rate_limit")
        views = [FP("a", [M("a", "m")]), FP("b", [M("b", "m")])]
        plan = plan_attempts(views, set(), set(), 0, {}, has_key=lambda p: True, health=h, recent={})
        self.assertEqual([a.provider.provider_id for a in plan], ["b", "a"])

    def test_too_small_context_goes_after_those_that_fit(self):
        views = [FP("a", [M("a", "small", ctx=4000)]), FP("b", [M("b", "big", ctx=1_000_000)])]
        plan = plan_attempts(views, set(), set(), 50_000, {}, has_key=lambda p: True, health=Health(), recent={})
        self.assertEqual(plan[0].model.model_id, "big")
        self.assertEqual(len(plan), 2)

    def test_input_capability_filtering(self):
        views = [FP("no", [M("no", "m", support={"image": False})]),
                 FP("yes", [M("yes", "m", support={"image": True})]),
                 FP("unk", [M("unk", "m")])]
        plan = plan_attempts(views, {"image"}, set(), 0, {}, has_key=lambda p: True, health=Health(), recent={})
        ids = [a.provider.provider_id for a in plan]
        self.assertNotIn("no", ids)               # provider said no -> dropped
        self.assertEqual(ids[0], "yes")           # confirmed first
        self.assertIn("unk", ids)                 # unknown is still tried

    def test_a_past_rejection_demotes_but_never_removes(self):
        now = datetime.now(timezone.utc).isoformat()
        checks = {("a", "m", "input:image"): {"result": False, "evidence": "x", "checked_at": now}}
        views = [FP("a", [M("a", "m")]), FP("b", [M("b", "m")])]
        plan = plan_attempts(views, {"image"}, set(), 0, checks, has_key=lambda p: True, health=Health(), recent={})
        self.assertEqual([a.provider.provider_id for a in plan], ["b", "a"])
        plan = plan_attempts([views[0]], {"image"}, set(), 0, checks, has_key=lambda p: True, health=Health(), recent={})
        self.assertEqual(len(plan), 1)  # the only model is still tried

    def test_providers_without_keys_are_skipped(self):
        views = [FP("a", [M("a", "m")]), FP("b", [M("b", "m")])]
        plan = plan_attempts(views, set(), set(), 0, {}, has_key=lambda p: p == "b", health=Health(), recent={})
        self.assertEqual([a.provider.provider_id for a in plan], ["b"])


class Fallback(unittest.TestCase):
    def views(self):
        return [FP("a", [M("a", "m1"), M("a", "m2")]), FP("b", [M("b", "m1")]), FP("c", [M("c", "m1")])]

    def test_every_kind_of_failure_moves_on(self):
        for exc in (Err("slow down", 429), Err("boom", 500), TimeoutError("timed out"), Err("bad key", 401),
                    Err("model not found", 404), Err("weird", 400), ValueError("anything at all")):
            h = Harness(self.views(), {("a", "m1"): exc, ("a", "m2"): exc, ("b", "m1"): exc})
            out = h.run()
            self.assertEqual(out.provider.provider_id, "c", repr(exc))
            self.assertIn("switched to", out.fallback_reason)

    def test_without_safe_fallback_a_failure_is_raised_and_nothing_else_is_tried(self):
        h = Harness([FP("a", [M("a", "m1")]), FP("b", [M("b", "m1")])], {("a", "m1"): Err("boom", 500), ("b", "m1"): Err("boom", 500)})
        with self.assertRaises(Err):
            h.run(safe_fallback=False, auto=False, selected=("a", "m1"))
        self.assertEqual(len(h.calls), 1)

    def test_manual_selection_tried_first_then_the_rest(self):
        h = Harness(self.views(), {("b", "m1"): Err("down", 503)})
        out = h.run(auto=False, selected=("b", "m1"))
        self.assertEqual(h.calls[0][:2], ("b", "m1"))
        self.assertNotEqual((out.provider.provider_id, out.model_id), ("b", "m1"))

    def test_bad_key_skips_the_rest_of_that_provider(self):
        h = Harness([FP("a", [M("a", f"m{i}") for i in range(5)]), FP("b", [M("b", "ok")])],
                    {("a", f"m{i}"): Err("invalid api key", 401) for i in range(5)})
        h.run(rng=random.Random(3))
        self.assertLessEqual(sum(1 for c in h.calls if c[0] == "a"), 1)

    def test_empty_reply_counts_as_a_failure_under_fallback(self):
        h = Harness([FP("a", [M("a", "m")]), FP("b", [M("b", "m")])], {("a", "m"): "", ("b", "m"): ""})
        h.behave[("b", "m")] = "real answer"
        h.behave[("a", "m")] = ""
        out = h.run()
        self.assertEqual(out.answer, "real answer")

    def test_budget_block_skips_to_a_candidate_that_fits(self):
        def gate(pid, mid, messages):
            if pid == "a":
                raise BudgetBlocked({"message": "over limit"})
        h = Harness([FP("a", [M("a", "m")]), FP("b", [M("b", "m")])])
        self.assertEqual(h.run(budget_gate=gate).provider.provider_id, "b")

    def test_attempt_cap(self):
        views = [FP(f"p{i}", [M(f"p{i}", "m")]) for i in range(6)]
        h = Harness(views, {(f"p{i}", "m"): Err("boom", 500) for i in range(6)})
        with self.assertRaises(AllAttemptsFailed) as cm:
            h.run(max_attempts=3)
        self.assertEqual(len(cm.exception.log), 3)

    def test_all_failed_reports_each_attempt(self):
        h = Harness([FP("a", [M("a", "m")]), FP("b", [M("b", "m")])], {("a", "m"): Err("x", 500), ("b", "m"): Err("y", 429)})
        with self.assertRaises(AllAttemptsFailed) as cm:
            h.run()
        self.assertEqual({e["kind"] for e in cm.exception.log}, {"server", "rate_limit"})

    def test_no_keys_explains(self):
        h = Harness([FP("a", [M("a", "m")])])
        with self.assertRaises(NoCandidates) as cm:
            h.run(has_key=lambda p: False)
        self.assertIn("API key", str(cm.exception))


class Learning(unittest.TestCase):
    def test_image_request_walks_to_a_model_that_works_and_remembers(self):
        views = [FP("blind", [M("blind", "m")]), FP("sees", [M("sees", "m", support={"image": True})])]
        h = Harness(views, {("blind", "m"): UnsupportedModality("no images", "image")})
        h.health.penalize("sees", "m", "rate_limit")  # make the unconfirmed model go first, deterministically
        out = h.run(prepared=image_prep())
        self.assertEqual([c[0] for c in h.calls], ["blind", "sees"])
        self.assertEqual(out.provider.provider_id, "sees")
        self.assertTrue(h.checks[("sees", "m", "input:image")]["result"])
        self.assertFalse(h.checks[("blind", "m", "input:image")]["result"])
        # next time the known-incapable model is not even a candidate
        h.health.clear()
        h.calls.clear()
        h.run(prepared=image_prep())
        self.assertNotIn("blind", [c[0] for c in h.calls])

    def test_a_provider_message_about_unsupported_images_is_learned_from(self):
        views = [FP("x", [M("x", "m")]), FP("y", [M("y", "m", support={"image": True})])]
        h = Harness(views, {("x", "m"): Err("This model does not support image input", 400)})
        h.health.penalize("y", "m", "rate_limit")
        h.run(prepared=image_prep())
        self.assertEqual(h.calls[0][0], "x")
        self.assertFalse(h.checks[("x", "m", "input:image")]["result"])

    def test_other_errors_do_not_poison_capability(self):
        views = [FP("x", [M("x", "m")]), FP("y", [M("y", "m", support={"image": True})])]
        h = Harness(views, {("x", "m"): Err("overloaded", 503)})
        h.health.penalize("y", "m", "rate_limit")
        h.run(prepared=image_prep())
        self.assertEqual([c[0] for c in h.calls], ["x", "y"])
        self.assertNotIn(("x", "m", "input:image"), h.checks)

    def test_media_reaches_the_model_as_blocks(self):
        h = Harness([FP("sees", [M("sees", "m", support={"image": True})])])
        h.run(prepared=image_prep())
        content = h.calls[0][2]
        self.assertTrue(any(isinstance(b, dict) and b["type"] == "media" and b["kind"] == "image" for b in content))
        self.assertEqual(base64.b64decode(next(b for b in content if b["type"] == "media")["data"]), PNG)

    def test_if_nothing_accepts_the_files_the_task_still_completes_from_text(self):
        views = [FP("a", [M("a", "m")]), FP("b", [M("b", "m")])]
        def behave(messages):
            if any(isinstance(b, dict) and b.get("type") == "media" for b in (messages[-1]["content"] if isinstance(messages[-1]["content"], list) else [])):
                raise UnsupportedModality("no images", "image")
            return "answered from text"
        h = Harness(views, {("a", "m"): behave, ("b", "m"): behave})
        prep = Prepared([Part("a.png", "image", "image/png", PNG), Part("n.txt", "text", "text/plain", None, "the notes")])
        out = h.run(prepared=prep)
        self.assertEqual(out.answer, "answered from text")
        self.assertTrue(out.degraded)
        self.assertEqual(out.skipped, ["a.png"])
        self.assertIn("a.png", out.fallback_reason)
        self.assertIn("the notes", str(h.calls[-1][2]))

    def test_text_only_model_gets_pdf_text_when_it_cannot_take_the_pdf(self):
        pdf = Part("t.pdf", "pdf", "application/pdf", b"%PDF", "words from the pdf")
        h = Harness([FP("a", [M("a", "m", support={"pdf": False})])])
        h.run(prepared=Prepared([pdf]))
        self.assertIn("words from the pdf", str(h.calls[0][2]))
        self.assertNotIn("'type': 'media'", str(h.calls[0][2]))

    def test_degrade_is_not_used_without_safe_fallback(self):
        h = Harness([FP("a", [M("a", "m", support={"image": False})])])
        with self.assertRaises(NoCandidates):
            h.run(prepared=image_prep(), safe_fallback=False)


if __name__ == "__main__":
    unittest.main()
