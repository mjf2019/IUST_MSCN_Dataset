import random
import unittest
from amcal_matched_budget import matched_query_positions


class MatchedBudget(unittest.TestCase):
    def test_exact_counts(self):
        labels = [0]*103+[1]*79+[2]*282+[3]*218+[4]*80+[5]*311
        caps = {'0':1,'1':0,'2':13,'3':2,'4':0,'5':24}
        chosen = matched_query_positions(labels,caps,42)
        self.assertEqual(len(chosen),40)
        for c,n in caps.items():
            self.assertEqual(sum(labels[i]==int(c) for i in chosen),n)

    def test_rng_and_reproducibility(self):
        before=random.getstate()
        a=matched_query_positions([0]*80,{0:16},42)
        self.assertEqual(random.getstate(),before)
        self.assertEqual(a,matched_query_positions([0]*80,{0:16},42))
        self.assertTrue(any(i>=40 for i in a))

    def test_zero_full_and_invalid(self):
        self.assertEqual(matched_query_positions([0,1],{0:0,1:1},42),{1})
        for caps in ({0:3,1:0},{0:-1,1:0},{0:1.5,1:0},{0:0}):
            with self.assertRaises(ValueError):
                matched_query_positions([0,1],caps,42)


if __name__=='__main__':
    unittest.main()
