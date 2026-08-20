import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SocraticUiContractTests(unittest.TestCase):
    def test_off_state_short_circuits_before_followup_request(self):
        hook = (ROOT / "src" / "hooks" / "useSocraticFollowups.ts").read_text(encoding="utf-8")
        guard = hook.index("if (!enabled || !response || !hasAnswer || !response.answer)")
        request = hook.index("void fetchSocraticFollowups")
        self.assertLess(guard, request)

    def test_on_state_calls_dedicated_endpoint_after_response(self):
        service = (ROOT / "src" / "services" / "qaService.ts").read_text(encoding="utf-8")
        hook = (ROOT / "src" / "hooks" / "useSocraticFollowups.ts").read_text(encoding="utf-8")
        self.assertIn("/api/socratic/followups", service)
        self.assertIn("response.answer", hook)
        self.assertIn("fetchSocraticFollowups", hook)

    def test_clicking_suggestion_submits_it_to_main_qa_pipeline(self):
        home = (ROOT / "src" / "pages" / "HomePage.tsx").read_text(encoding="utf-8")
        self.assertIn("submitQuestion(followUp.question)", home)
        self.assertIn("onFollowUpSelect={askFollowUp}", home)

    def test_followup_failure_does_not_replace_main_answer(self):
        hook = (ROOT / "src" / "hooks" / "useSocraticFollowups.ts").read_text(encoding="utf-8")
        error_branch = hook[hook.index(".catch((error: unknown)") :]
        self.assertIn("setFollowUps([])", error_branch)
        self.assertIn("setLoadState('error')", error_branch)
        self.assertNotIn("setAnswer", error_branch)

    def test_toggle_and_chips_are_keyboard_accessible(self):
        toggle = (ROOT / "src" / "components" / "socratic" / "SocraticToggle.tsx").read_text(encoding="utf-8")
        chip = (ROOT / "src" / "components" / "socratic" / "FollowUpChip.tsx").read_text(encoding="utf-8")
        self.assertIn('role="switch"', toggle)
        self.assertIn("aria-checked={enabled}", toggle)
        self.assertIn("absolute left-1 top-1", toggle)
        self.assertIn("translate-x-5", toggle)
        self.assertGreaterEqual(chip.count('<button'), 1)
        self.assertIn("aria-label", chip)

    def test_socratic_preference_defaults_off_and_is_persisted(self):
        store = (ROOT / "src" / "store" / "appStore.ts").read_text(encoding="utf-8")
        self.assertIn("socraticEnabled: false", store)
        self.assertIn("settings: state.settings", store)

    def test_voice_and_draft_question_are_deduplicated_before_submit(self):
        home = (ROOT / "src" / "pages" / "HomePage.tsx").read_text(encoding="utf-8")
        input_component = (
            ROOT / "src" / "components" / "input" / "QuestionInput.tsx"
        ).read_text(encoding="utf-8")
        pipeline = (ROOT / "src" / "hooks" / "useQaPipeline.ts").read_text(encoding="utf-8")
        utility = (ROOT / "src" / "utils" / "questionInput.ts").read_text(encoding="utf-8")

        self.assertIn("mergeQuestionParts(draft, speech.transcript", home)
        self.assertIn("mergeQuestionParts(value, transcript, interimTranscript)", input_component)
        self.assertIn("collapseRepeatedQuestion(question)", pipeline)
        self.assertIn("normalizedQuestion(first) === normalizedQuestion(second)", utility)

    def test_question_input_preserves_spaces_and_enter_submits(self):
        input_component = (
            ROOT / "src" / "components" / "input" / "QuestionInput.tsx"
        ).read_text(encoding="utf-8")
        home = (ROOT / "src" / "pages" / "HomePage.tsx").read_text(encoding="utf-8")

        self.assertIn(": value;", input_component)
        self.assertIn("onKeyDown={handleKeyDown}", input_component)
        self.assertIn("event.key !== 'Enter'", input_component)
        self.assertIn("event.shiftKey", input_component)
        self.assertIn("event.nativeEvent.isComposing", input_component)
        self.assertIn("event.preventDefault()", input_component)
        self.assertIn("onChange={changeDraft}", home)
        self.assertIn("speech.resetTranscript()", home)

    def test_zero_followups_has_no_production_empty_shell(self):
        insights = (ROOT / "src" / "components" / "socratic" / "FollowUpInsights.tsx").read_text(encoding="utf-8")
        self.assertIn("loadState === 'ready' && followUps.length === 0 && !showDebug", insights)
        self.assertIn("Chưa có gợi ý phù hợp", insights)

    def test_one_or_three_followups_render_exactly_available_chips(self):
        insights = (ROOT / "src" / "components" / "socratic" / "FollowUpInsights.tsx").read_text(encoding="utf-8")
        self.assertIn("followUps.map((followUp)", insights)
        self.assertNotIn("placeholder", insights.casefold())
        self.assertNotIn("followUps.length < 3", insights)

    def test_loading_copy_mentions_mari_and_cannot_persist_after_ready(self):
        insights = (ROOT / "src" / "components" / "socratic" / "FollowUpInsights.tsx").read_text(encoding="utf-8")
        self.assertIn("Mari đang tìm hướng khám phá tiếp", insights)
        self.assertIn("loadState === 'loading'", insights)


if __name__ == "__main__":
    unittest.main()
