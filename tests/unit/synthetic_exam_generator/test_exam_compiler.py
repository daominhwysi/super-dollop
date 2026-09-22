import unittest
from unittest.mock import patch
from synthetic_exam_generator.exam_compiler import (
    generate_exam_tasks,
    generate_single_exam,
    SECTION_MC,
    SECTION_TF,
    SECTION_SA,
    SECTION_ESSAY,
    SECTION_LIT_READING,
    SECTION_LIT_WRITING
)
from synthetic_exam_generator.generator import Subject

class TestExamCompiler(unittest.TestCase):
    def test_generate_exam_tasks(self):
        tasks = generate_exam_tasks()
        sections = [t[0] for t in tasks]
        
        mc_count = sections.count(SECTION_MC)
        self.assertTrue(10 <= mc_count <= 20)
        
        tf_count = sections.count(SECTION_TF)
        self.assertTrue(1 <= tf_count <= 6)
        
        sa_count = sections.count(SECTION_SA)
        self.assertTrue(0 <= sa_count <= 6)
        
        essay_count = sections.count(SECTION_ESSAY)
        self.assertTrue(0 <= essay_count <= 4)

    @patch('synthetic_exam_generator.exam_compiler.generate_single_question')
    def test_generate_single_exam(self, mock_gen):
        mock_gen.return_value = {
            "stem": "Dummy question stem",
            "options": ["A", "B", "C", "D"],
            "raw_text": "Dummy raw text",
            "spans": [{"start": 0, "end": 5}],
            "problem_type_level": "VD"
        }
        
        exam_data = generate_single_exam(
            subject=Subject.CHEMISTRY,
            grade=10,
            concurrency=2
        )
        
        self.assertIsNotNone(exam_data)
        self.assertEqual(exam_data["subject"], "chemistry")
        self.assertEqual(exam_data["grade"], 10)
        self.assertIn("exam_id", exam_data)
        
        sections = exam_data["sections"]
        self.assertIn(SECTION_MC, sections)
        self.assertIn(SECTION_TF, sections)
        self.assertIn(SECTION_SA, sections)
        
        mc_questions = sections[SECTION_MC]
        self.assertTrue(len(mc_questions) >= 10)
        for q in mc_questions:
            self.assertNotIn("raw_text", q)
            self.assertNotIn("spans", q)
            self.assertNotIn("problem_type_level", q)

    @patch('synthetic_exam_generator.exam_compiler.generate_single_question')
    def test_generate_single_exam_english(self, mock_gen):
        mock_gen.return_value = {
            "stem": "Dummy English stem",
            "options": ["A", "B", "C", "D"],
            "raw_text": "Dummy raw text",
            "spans": [{"start": 0, "end": 5}],
            "problem_type_level": "VD"
        }
        
        exam_data = generate_single_exam(
            subject=Subject.ENGLISH,
            grade=12,
            concurrency=2
        )
        
        self.assertIsNotNone(exam_data)
        self.assertEqual(exam_data["subject"], "english")
        self.assertEqual(exam_data["grade"], 12)
        
        sections = exam_data["sections"]
        self.assertTrue(len(sections) > 0)
        
        # Verify randomized English tasks count matches either New (10) or Old (36) format task counts
        total_questions = sum(len(q_list) for q_list in sections.values())
        self.assertIn(total_questions, {15, 36})

    @patch('synthetic_exam_generator.exam_compiler.generate_single_question')
    def test_generate_single_exam_literature(self, mock_gen):
        mock_gen.return_value = {
            "stem": "Dummy Literature stem",
            "options": [],
            "raw_text": "Dummy raw text",
            "spans": [{"start": 0, "end": 5}],
            "problem_type_level": "VD"
        }
        
        exam_data = generate_single_exam(
            subject=Subject.LITERATURE,
            grade=11,
            concurrency=2
        )
        
        self.assertIsNotNone(exam_data)
        self.assertEqual(exam_data["subject"], "literature")
        self.assertEqual(exam_data["grade"], 11)
        
        sections = exam_data["sections"]
        self.assertEqual(len(sections), 2)
        total_questions = sum(len(q_list) for q_list in sections.values())
        self.assertTrue(2 <= total_questions <= 3)

    @patch('synthetic_exam_generator.exam_compiler.generate_single_question')
    def test_generate_single_exam_toeic(self, mock_gen):
        mock_gen.return_value = {
            "stem": "Ms. Clara requested that the report be delivered _____ Friday.",
            "options": ["(A) by", "(B) on", "(C) at", "(D) with"],
            "raw_text": "Dummy TOEIC raw text",
            "spans": [{"start": 0, "end": 5}],
            "problem_type_level": "VD"
        }
        
        exam_data = generate_single_exam(
            subject=Subject.TOEIC,
            grade=12,
            concurrency=2
        )
        
        self.assertIsNotNone(exam_data)
        self.assertEqual(exam_data["subject"], "toeic")
        self.assertEqual(exam_data["grade"], 12)
        
        sections = exam_data["sections"]
        self.assertIn("PART 5 - INCOMPLETE SENTENCES", sections)
        self.assertIn("PART 7 - READING COMPREHENSION", sections)

    @patch('synthetic_exam_generator.exam_compiler.generate_single_exam')
    @patch('synthetic_exam_generator.exam_compiler.get_available_curricula')
    def test_run_batch_exams_generator(self, mock_get_curr, mock_gen_exam):
        from synthetic_exam_generator.exam_compiler import run_batch_exams_generator
        from tempfile import TemporaryDirectory
        import os
        
        mock_get_curr.return_value = [("physics", 11)]
        import uuid
        mock_gen_exam.side_effect = lambda subject, grade, **kwargs: {
            "exam_id": uuid.uuid4().hex[:8],
            "subject": subject.value,
            "grade": grade,
            "created_at": "2026-06-02T00:00:00",
            "sections": {}
        }
        
        with TemporaryDirectory() as tmpdir:
            run_batch_exams_generator(
                num_exams=2,
                output_dir=tmpdir,
                subject="physics",
                grade=11
            )
            
            # Check that files were written
            files = os.listdir(tmpdir)
            self.assertEqual(len(files), 2)
            for f in files:
                self.assertTrue(f.startswith("exam_physics_g11_"))

if __name__ == '__main__':
    unittest.main()
