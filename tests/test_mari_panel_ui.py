import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MariPanelUiTests(unittest.TestCase):
    def read(self, relative: str) -> str:
        return (ROOT / relative).read_text(encoding="utf-8")

    def test_mari_panel_renders_existing_vrm_scene(self):
        panel = self.read("src/components/avatar/MariPanel.tsx")
        self.assertIn("<AvatarScene state={state} compact", panel)
        self.assertIn('aria-label="Mari 3D assistant"', panel)
        self.assertIn("Trợ lý học tập VIQA", panel)
        self.assertIn("Gia sư luôn bật", panel)

    def test_desktop_layout_contains_collapsible_avatar_column(self):
        home = self.read("src/pages/HomePage.tsx")
        panel = self.read("src/components/avatar/MariPanel.tsx")
        self.assertIn("min-[1200px]:grid-cols-[248px_minmax(0,1fr)_272px]", home)
        self.assertIn("collapsed={avatarCollapsed}", home)
        self.assertIn("pointer-events-none invisible opacity-0", panel)
        self.assertIn("w-[272px]", panel)

    def test_mobile_uses_one_on_demand_avatar_sheet(self):
        home = self.read("src/pages/HomePage.tsx")
        header = self.read("src/components/layout/Header.tsx")
        self.assertIn("!desktopAvatarVisible && mobileMariOpen", home)
        self.assertIn("<MariSheet", home)
        self.assertIn('aria-label="Hiện Mari 3D"', header)
        self.assertNotIn("<AvatarScene", home)

    def test_existing_animation_and_lipsync_implementation_remains_connected(self):
        avatar = self.read("src/components/avatar/AnimeAvatar.tsx")
        home = self.read("src/pages/HomePage.tsx")
        self.assertIn("AvatarLookAtController", avatar)
        self.assertIn("AvatarHairMotionController", avatar)
        self.assertIn("AvatarPoseController", avatar)
        self.assertIn("state === 'speaking'", avatar)
        self.assertIn("synthesis.speaking", home)

    def test_model_loading_and_error_states_are_visible(self):
        avatar = self.read("src/components/avatar/AnimeAvatar.tsx")
        self.assertIn("/models/mari.vrm", avatar)
        self.assertIn("Đang tải Mari...", avatar)
        self.assertIn("Không thể tải avatar Mari.", avatar)


if __name__ == "__main__":
    unittest.main()
