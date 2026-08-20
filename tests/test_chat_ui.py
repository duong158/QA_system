import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ChatUiContractTests(unittest.TestCase):
    def read(self, relative: str) -> str:
        return (ROOT / relative).read_text(encoding="utf-8")

    def test_home_uses_three_column_conversation_shell(self):
        home = self.read("src/pages/HomePage.tsx")
        self.assertIn("lg:grid-cols-[248px_minmax(0,1fr)]", home)
        self.assertIn("min-[1200px]:grid-cols-[248px_minmax(0,1fr)_272px]", home)
        self.assertIn("<ChatSidebar", home)
        self.assertIn("<ChatComposer", home)
        self.assertIn("<MariPanel", home)

    def test_conversation_keeps_independent_turns(self):
        home = self.read("src/pages/HomePage.tsx")
        chat_type = self.read("src/types/chat.ts")
        self.assertIn("const [turns, setTurns]", home)
        self.assertIn("turns.map((turn)", home)
        self.assertIn("status: 'pending'", home)
        self.assertIn("response?: QaResponse", chat_type)

    def test_composer_preserves_spaces_and_enter_contract(self):
        composer = self.read("src/components/chat/ChatComposer.tsx")
        self.assertIn(": value;", composer)
        self.assertIn("event.key !== 'Enter'", composer)
        self.assertIn("event.shiftKey", composer)
        self.assertIn("event.nativeEvent.isComposing", composer)
        self.assertIn("event.preventDefault()", composer)
        self.assertIn("displayValue.trim()", composer)
        self.assertIn("textarea.style.height", composer)

    def test_source_and_pipeline_are_collapsed_disclosures(self):
        message = self.read("src/components/chat/AssistantMessage.tsx")
        self.assertIn("<details", message)
        self.assertIn(">Nguồn</span>", message)
        self.assertIn("Xem cách hệ thống tìm câu trả lời", message)
        self.assertIn("showDebug && selected", message)
        self.assertIn("Tín hiệu xếp hạng ứng viên, không phải xác suất đúng", message)

    def test_socratic_suggestions_are_conversational_quick_replies(self):
        home = self.read("src/pages/HomePage.tsx")
        chip = self.read("src/components/socratic/FollowUpChip.tsx")
        self.assertIn("await submitConversationQuestion(followUp.question)", home)
        self.assertIn("bg-violet-50", chip)
        self.assertIn("onSelect(followUp)", chip)

    def test_thinking_and_guarded_auto_scroll_exist(self):
        home = self.read("src/pages/HomePage.tsx")
        message = self.read("src/components/chat/AssistantMessage.tsx")
        self.assertIn("shouldAutoScrollRef", home)
        self.assertIn("viewport.scrollHeight - viewport.scrollTop", home)
        self.assertIn("Mari đang suy nghĩ", message)
        self.assertIn("role=\"status\"", message)

    def test_light_tokens_and_dark_palette_are_declared(self):
        css = self.read("src/index.css")
        for token in ("--background", "--surface", "--border", "--text-primary", "--primary", "--socratic"):
            self.assertIn(token, css)
        self.assertIn("[data-theme='dark']", css)
        self.assertNotIn("fonts.googleapis.com", css)

    def test_mobile_history_drawer_and_avatar_breakpoint(self):
        sidebar = self.read("src/components/chat/ChatSidebar.tsx")
        home = self.read("src/pages/HomePage.tsx")
        self.assertIn("fixed inset-0 z-50 lg:hidden", sidebar)
        self.assertIn("<MariSheet", home)
        self.assertIn("window.matchMedia('(min-width: 1200px)')", home)
        self.assertIn("lg:hidden", sidebar)

    def test_feedback_copy_speak_and_source_actions_are_available(self):
        message = self.read("src/components/chat/AssistantMessage.tsx")
        self.assertIn("<AnswerFeedback response={response} compact", message)
        self.assertIn("navigator.clipboard.writeText", message)
        self.assertIn("onSpeakAnswer(answerText)", message)

    def test_blind_spot_dashboard_remains_secondary_route(self):
        sidebar = self.read("src/components/chat/ChatSidebar.tsx")
        dashboard = self.read("src/pages/KnowledgeBlindSpotsPage.tsx")
        self.assertIn('to="/knowledge-blind-spots"', sidebar)
        self.assertIn('id="document-contribution"', dashboard)


if __name__ == "__main__":
    unittest.main()
