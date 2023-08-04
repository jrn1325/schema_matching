import unittest
import csm

class TestQuadraticProgram(unittest.TestCase):

    def setUp(self):
        target_file = "files/target_file.json"
        source_file = "files/source_file.json"
        self.df_target = csm.json_to_dataframe(target_file)
        self.df_source = csm.json_to_dataframe(source_file)
  
    
    def _test_match(self, ground_truth, correct_match_dict):
        match_dict = csm.find_valentine(self.df_target, self.df_source, ground_truth)
        final_match_dict = csm.quadratic_programming(match_dict)
        self.assertEqual(final_match_dict, correct_match_dict)
    
    def test_first_last(self):
        ground_truth = [("t_name.firstname", "s_name.firstname"), ("t_name.lastname", "s_name.firstname")]
        correct_match_dict = {"s_name.firstname": "t_name.firstname"}
        self._test_match(ground_truth, correct_match_dict)

    def test_first_last_age(self):
        ground_truth = [("t_name.firstname", "s_name.firstname"), ("t_name.lastname", "s_name.firstname"), ("t_age", "s_age")]
        correct_match_dict = {"s_name.firstname": "t_name.firstname", "s_age": "t_age"}
        self._test_match(ground_truth, correct_match_dict)

    def test_first_last_age_gender(self):
        ground_truth = [("t_name.firstname", "s_name.firstname"), ("t_name.lastname", "s_name.firstname"), ("t_age", "s_age"), ("t_gender", "s_gender")]
        correct_match_dict = {"s_name.firstname": "t_name.firstname", "s_age": "t_age", "s_gender": "t_gender"}
        self._test_match(ground_truth, correct_match_dict)
    

if __name__ == '__main__':
    unittest.main()