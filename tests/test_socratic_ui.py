import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SocraticUiContractTests(unittest.TestCase):
    def test_followups_run_automatically_after_answer(self):
        hook = (ROOT / "src" / "hooks" / "useSocraticFollowups.ts").read_text(encoding="utf-8")
        guard = hook.index("if (!response || !hasContext)")
        request = hook.index("void fetchSocraticFollowups")
        self.assertLess(guard, request)
        self.assertIn("useSocraticFollowups(response: QaResponse | null)", hook)
        self.assertNotIn("!enabled", hook)
        self.assertNotIn("!response.answer", hook)
        self.assertIn("contextPassageIds: Set<string>", hook)
        self.assertIn("retrieved_passage_ids: Array.from(sessionRef.current.contextPassageIds)", hook)

    def test_on_state_calls_dedicated_endpoint_after_response(self):
        service = (ROOT / "src" / "services" / "qaService.ts").read_text(encoding="utf-8")
        hook = (ROOT / "src" / "hooks" / "useSocraticFollowups.ts").read_text(encoding="utf-8")
        self.assertIn("/api/socratic/followups", service)
        self.assertIn("response.answer", hook)
        self.assertIn("fetchSocraticFollowups", hook)

    def test_clicking_suggestion_submits_it_to_main_qa_pipeline(self):
        home = (ROOT / "src" / "pages" / "HomePage.tsx").read_text(encoding="utf-8")
        pipeline = (ROOT / "src" / "hooks" / "useQaPipeline.ts").read_text(encoding="utf-8")
        qa_types = (ROOT / "src" / "types" / "qa.ts").read_text(encoding="utf-8")
        api = (ROOT / "backend" / "viqa_api.py").read_text(encoding="utf-8")
        self.assertIn("submitConversationQuestion(followUp.question, followUp.source_passage_id)", home)
        self.assertIn("preferred_passage_id: options.preferredPassageId", pipeline)
        self.assertIn("grounded_passage_only: Boolean(options.preferredPassageId)", pipeline)
        self.assertIn("preferred_passage_id?: string | null", qa_types)
        self.assertIn("answerability_validator=_validate_socratic_answerability", api)
        self.assertIn("onFollowUpSelect={askFollowUp}", home)

    def test_followup_failure_does_not_replace_main_answer(self):
        hook = (ROOT / "src" / "hooks" / "useSocraticFollowups.ts").read_text(encoding="utf-8")
        error_branch = hook[hook.index(".catch((error: unknown)") :]
        self.assertIn("setFollowUps([])", error_branch)
        self.assertIn("setLoadState('error')", error_branch)
        self.assertNotIn("setAnswer", error_branch)

    def test_toggle_is_removed_and_chips_remain_keyboard_accessible(self):
        home = (ROOT / "src" / "pages" / "HomePage.tsx").read_text(encoding="utf-8")
        header = (ROOT / "src" / "components" / "layout" / "Header.tsx").read_text(encoding="utf-8")
        composer = (ROOT / "src" / "components" / "chat" / "ChatComposer.tsx").read_text(encoding="utf-8")
        chip = (ROOT / "src" / "components" / "socratic" / "FollowUpChip.tsx").read_text(encoding="utf-8")
        self.assertFalse((ROOT / "src" / "components" / "socratic" / "SocraticToggle.tsx").exists())
        self.assertNotIn("socraticEnabled", home)
        self.assertNotIn('role="switch"', header)
        self.assertNotIn('role="switch"', composer)
        self.assertGreaterEqual(chip.count('<button'), 1)
        self.assertIn("aria-label", chip)

    def test_socratic_preference_is_removed_from_persisted_settings(self):
        store = (ROOT / "src" / "store" / "appStore.ts").read_text(encoding="utf-8")
        hook = (ROOT / "src" / "hooks" / "useSocraticFollowups.ts").read_text(encoding="utf-8")
        self.assertNotIn("socraticEnabled: false", store)
        self.assertIn("delete migratedSettings.socraticEnabled", store)
        self.assertIn("const socratic = useSocraticFollowups(answer)", (ROOT / "src" / "pages" / "HomePage.tsx").read_text(encoding="utf-8"))
        self.assertNotIn("enabled: boolean", hook)
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

    def test_development_mode_requests_and_renders_socratic_diagnostics(self):
        hook = (ROOT / "src" / "hooks" / "useSocraticFollowups.ts").read_text(encoding="utf-8")
        insights = (
            ROOT / "src" / "components" / "socratic" / "FollowUpInsights.tsx"
        ).read_text(encoding="utf-8")
        home = (ROOT / "src" / "pages" / "HomePage.tsx").read_text(encoding="utf-8")
        self.assertIn("debug: import.meta.env.DEV", hook)
        self.assertIn("setDebug(result.debug ?? null)", hook)
        self.assertIn("Opportunities found:", insights)
        self.assertIn("Candidates generated:", insights)
        self.assertIn("debug.rejection_distribution", insights)
        self.assertIn("followUpsDebug={isCurrentResponse ? socratic.debug : null}", home)


if __name__ == "__main__":
    unittest.main()
