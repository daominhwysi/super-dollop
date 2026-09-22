import unittest
import unittest.mock
import os
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from synthetic_exam_generator.curriculum import (
    load_curriculum,
    select_curriculum_path,
    map_cognitive_level_to_difficulty,
    get_curriculum_path
)

class TestCurriculum(unittest.TestCase):
    def setUp(self):
        # Create a sample curriculum configuration for testing
        self.sample_curriculum = {
            "subject": "physics",
            "grade": 11,
            "chapters": [
                {
                    "name": "Sóng",
                    "units": [
                        {
                            "name": "Sóng ánh sáng",
                            "problem_types": [
                                {
                                    "id": "physics_11_wave_light_wave_dang1",
                                    "name": "Lý thuyết về các loại bức xạ và quang phổ",
                                    "cognitive_level": "NB_TH",
                                    "details": "Bản chất sóng điện từ...",
                                    "examples": ["Ví dụ 1"]
                                },
                                {
                                    "id": "physics_11_wave_light_wave_dang4",
                                    "name": "Xác định số lượng vân sáng, vân tối",
                                    "cognitive_level": "VD",
                                    "details": "Trường đối xứng...",
                                    "examples": []
                                }
                            ]
                        },
                        {
                            "name": "Sóng âm",
                            "problem_types": [
                                {
                                    "id": "physics_11_wave_sound_wave_dang1",
                                    "name": "Đặc trưng vật lý của âm",
                                    "cognitive_level": "NB_TH",
                                    "details": "Tần số, cường độ âm...",
                                    "examples": []
                                }
                            ]
                        }
                    ]
                },
                {
                    "name": "Điện tích",
                    "units": [
                        {
                            "name": "Thuyết electron",
                            "problem_types": [
                                {
                                    "id": "physics_11_electrostatics_electron_dang1",
                                    "name": "Định luật Coulomb",
                                    "cognitive_level": "VD",
                                    "details": "Công thức F = k*q1*q2/r^2",
                                    "examples": []
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        
    def test_cognitive_level_mapping(self):
        # NB_TH should map to recognize or comprehend
        diff1 = map_cognitive_level_to_difficulty("NB_TH")
        self.assertIn(diff1, ["recognize", "comprehend"])
        
        # VD should map to low_application or application
        diff2 = map_cognitive_level_to_difficulty("VD")
        self.assertIn(diff2, ["low_application", "application"])
        
        # VDC should map to high_application
        diff3 = map_cognitive_level_to_difficulty("VDC")
        self.assertEqual(diff3, "high_application")
        
        # Unknown should fallback
        diff_unknown = map_cognitive_level_to_difficulty("UNKNOWN")
        self.assertIn(diff_unknown, ["recognize", "comprehend", "low_application", "application", "high_application"])

    def test_select_curriculum_path_no_filters(self):
        result = select_curriculum_path(self.sample_curriculum)
        self.assertIsNotNone(result)
        chapter, unit, pt = result
        self.assertIn("name", chapter)
        self.assertIn("name", unit)
        self.assertIn("id", pt)

    def test_select_curriculum_path_chapter_filter(self):
        # Case-insensitive substring match
        result = select_curriculum_path(self.sample_curriculum, chapter_filter="điện")
        self.assertIsNotNone(result)
        chapter, unit, pt = result
        self.assertEqual(chapter["name"], "Điện tích")
        self.assertEqual(unit["name"], "Thuyết electron")
        self.assertEqual(pt["id"], "physics_11_electrostatics_electron_dang1")
        
        # No match case
        no_match = select_curriculum_path(self.sample_curriculum, chapter_filter="nonexistent")
        self.assertIsNone(no_match)

    def test_select_curriculum_path_unit_filter(self):
        result = select_curriculum_path(self.sample_curriculum, unit_filter="âm")
        self.assertIsNotNone(result)
        chapter, unit, pt = result
        self.assertEqual(chapter["name"], "Sóng")
        self.assertEqual(unit["name"], "Sóng âm")
        self.assertEqual(pt["id"], "physics_11_wave_sound_wave_dang1")

    def test_select_curriculum_path_problem_type_filter(self):
        # Filter by ID
        result = select_curriculum_path(self.sample_curriculum, problem_type_filter="dang4")
        self.assertIsNotNone(result)
        chapter, unit, pt = result
        self.assertEqual(pt["id"], "physics_11_wave_light_wave_dang4")
        
        # Filter by Name
        result_name = select_curriculum_path(self.sample_curriculum, problem_type_filter="Coulomb")
        self.assertIsNotNone(result_name)
        _, _, pt_coulomb = result_name
        self.assertEqual(pt_coulomb["id"], "physics_11_electrostatics_electron_dang1")

    def test_load_curriculum_not_found_no_autogenerate(self):
        # Load from nonexistent file without autogenerate
        result = load_curriculum("nonexistent_subj", 99, autogenerate=False)
        self.assertIsNone(result)

    def test_fix_json_strings(self):
        from synthetic_exam_generator.curriculum import fix_json_strings
        
        # 1. Test escaping single backslashes in LaTeX blocks inside strings
        raw_json_latex = '{"details": "Công thức $v = \\frac{c}{n}$ và $\\lambda = i*a/D$"}' # here the escapes are single backslashes in raw text
        # in python string representing JSON:
        json_with_single = '{"details": "Công thức $v = \\frac{c}{n}$ và $\\lambda = i*a/D$"}'
        # If double backslash is already there:
        json_with_double = '{"details": "Công thức $v = \\\\frac{c}{n}$"}'
        
        fixed_single = fix_json_strings(json_with_single)
        # Verify that single backslashes inside $...$ are double escaped:
        self.assertIn('\\\\frac', fixed_single)
        self.assertIn('\\\\lambda', fixed_single)
        
        fixed_double = fix_json_strings(json_with_double)
        # Verify that double backslashes are not corrupted:
        self.assertIn('\\\\frac', fixed_double)
        
        # 2. Test literal newlines inside strings are escaped
        json_with_newline = '{"details": "Line 1\nLine 2"}'
        fixed_newline = fix_json_strings(json_with_newline)
        self.assertIn('Line 1\\nLine 2', fixed_newline)
        
        # 3. Test backslash newline scenario
        json_backslash_newline = '{"details": "Line 1\\\nLine 2"}'
        fixed_bs_newline = fix_json_strings(json_backslash_newline)
        self.assertIn('Line 1\\\\\\nLine 2', fixed_bs_newline)

    @unittest.mock.patch('synthetic_exam_generator.curriculum.chat')
    def test_generate_curriculum_passes_model_and_thinking(self, mock_chat):
        mock_chat.return_value = '{"subject": "physics", "grade": 11, "chapters": []}'
        from synthetic_exam_generator.curriculum import generate_curriculum
        
        with TemporaryDirectory() as tmpdir:
            with unittest.mock.patch('synthetic_exam_generator.curriculum.get_curriculum_path') as mock_path:
                mock_path.return_value = Path(tmpdir) / "test_subj_11.json"
                generate_curriculum("test_subj", 11, model="custom-model", thinking=True)
                
                mock_chat.assert_called_once()
                kwargs = mock_chat.call_args[1]
                self.assertEqual(kwargs.get("model"), "custom-model")
                self.assertEqual(kwargs.get("thinking"), True)

    @unittest.mock.patch('synthetic_exam_generator.generator.chat')
    @unittest.mock.patch('synthetic_exam_generator.generator.load_curriculum')
    def test_generate_single_question_passes_model_and_thinking(self, mock_load, mock_chat):
        from synthetic_exam_generator.generator import generate_single_question, Subject
        
        mock_load.return_value = {
            "subject": "physics",
            "grade": 11,
            "chapters": [
                {
                    "name": "Sóng",
                    "units": [
                        {
                            "name": "Sóng ánh sáng",
                            "problem_types": [
                                {
                                    "id": "physics_11_wave_light_wave_dang1",
                                    "name": "Lý thuyết",
                                    "cognitive_level": "NB_TH",
                                    "details": "Lý thuyết...",
                                    "examples": []
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        
        mock_chat.return_value = "<question><stem>Test stem</stem><option>Opt A</option></question>"
        
        generate_single_question(
            subject=Subject.PHYSICS,
            grade=11,
            model="custom-model-2",
            thinking=True
        )
        
        mock_load.assert_called_with('physics', 11, autogenerate=True, model='custom-model-2', thinking=True, provider=None)
        
        mock_chat.assert_called_once()
        kwargs = mock_chat.call_args[1]
        self.assertEqual(kwargs.get("model"), "custom-model-2")
        self.assertEqual(kwargs.get("thinking"), True)

    @unittest.mock.patch('synthetic_exam_generator.deepseek_client.client.chat.completions.create')
    def test_chat_thinking_modes(self, mock_create):
        from synthetic_exam_generator.deepseek_client import chat
        
        mock_response = unittest.mock.MagicMock()
        mock_response.choices = [unittest.mock.MagicMock()]
        mock_response.choices[0].message.content = "response"
        mock_response.model = "deepseek-v4-flash"
        
        mock_usage = unittest.mock.MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.completion_tokens_details = unittest.mock.MagicMock()
        mock_usage.completion_tokens_details.reasoning_tokens = 5
        mock_response.usage = mock_usage
        
        mock_create.return_value = mock_response
        
        chat("prompt", provider="deepseek", model="deepseek-v4-flash")
        mock_create.assert_called_with(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "prompt"}
            ],
            stream=False
        )
        
        mock_create.reset_mock()
        chat("prompt", thinking=True, provider="deepseek", model="deepseek-v4-flash")
        mock_create.assert_called_with(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "prompt"}
            ],
            stream=False,
            reasoning_effort="high"
        )
        
        mock_create.reset_mock()
        chat("prompt", thinking="max", provider="deepseek", model="deepseek-v4-flash")
        mock_create.assert_called_with(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "prompt"}
            ],
            stream=False,
            reasoning_effort="max"
        )
        
        mock_create.reset_mock()
        chat("prompt", thinking="none", provider="deepseek", model="deepseek-v4-flash")
        mock_create.assert_called_with(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "prompt"}
            ],
            stream=False
        )

if __name__ == '__main__':
    unittest.main()
